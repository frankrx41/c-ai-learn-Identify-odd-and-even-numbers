#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

// tick 554594
//      55187 * (34)

#pragma warning(disable:4996)

#define TRAIN_AI_NUM    ( 200 )
#define KEEP_TRAINED_AI ( 12 )      // TODO: find (x**2-x)*.5 = TRAIN_AI_NUM

#define INPUT_LEVEL_CELL_INDEX_MAX  ( 10*3 )
#define ANS_LEVEL_CELL_INDEX_MAX    ( 1 )
#define CELL_INDEX_MAX              ( ((INPUT_LEVEL_CELL_INDEX_MAX)*(INPUT_LEVEL_CELL_INDEX_MAX)-(INPUT_LEVEL_CELL_INDEX_MAX))/2 ) /* !!! Can not use *.5f */
#define CELL_LEVEL_MAX              ( 2 ) // ( CELL_INDEX_MAX + 1 )

#define CELL_ALL_INDEX              ( (INPUT_LEVEL_CELL_INDEX_MAX) + (CELL_LEVEL_MAX)*(CELL_INDEX_MAX) + (ANS_LEVEL_CELL_INDEX_MAX))

#define INPUT_LEVEL_WEIGHT_INDEX_MAX    ( 2 * (CELL_INDEX_MAX) )
#define ANS_LEVEL_WEIGHT_INDEX_MAX      ( (ANS_LEVEL_CELL_INDEX_MAX) * (CELL_INDEX_MAX) )
#define WEIGHT_INDEX_MAX                ( (CELL_INDEX_MAX) * (CELL_INDEX_MAX) )
#define WEIGHT_LEVEL_MAX                ( CELL_LEVEL_MAX - 1 )

#define WEIGHT_ALL_INDEX                ( (INPUT_LEVEL_WEIGHT_INDEX_MAX) + (WEIGHT_LEVEL_MAX)*(WEIGHT_INDEX_MAX) + (ANS_LEVEL_WEIGHT_INDEX_MAX))

typedef struct WeightTable
{
    union
    {
        struct
        {
            float input_level_weight[INPUT_LEVEL_WEIGHT_INDEX_MAX];
            float proc_level_weight[WEIGHT_LEVEL_MAX][WEIGHT_INDEX_MAX];
            float ans_level_weight[ANS_LEVEL_WEIGHT_INDEX_MAX];
        };
        float all_weight[WEIGHT_ALL_INDEX];
    };
}WeightTable;

typedef struct CellTable
{
    union
    {
        struct
        {
            float input_level_cell[INPUT_LEVEL_CELL_INDEX_MAX]; // here can change to bool, but for easy to use union, keep float
            float proc_cell[CELL_LEVEL_MAX][CELL_INDEX_MAX];
            float ans_level_cell[ANS_LEVEL_CELL_INDEX_MAX];     // bool
        };
        float all_cell[CELL_ALL_INDEX];
    };
}CellTable;

#if /* Tools */ 1
// [min,max) step = 0.001
float GetRandomFloat(const float min, const float max, const float step)
{
    if( min == max )
    {
        return min;
    }
    if( min < max )
    {
        const float stepInv = 1.f / step;
        const int length = (int)((max - min) * stepInv);
        return rand()%length/stepInv + min;
    }
    else
    {
        assert(!"Not support!");
        return 0;
    }
}

int GetRandomInt(const int min, const int max)
{
    if( min == max )
    {
        return min;
    }
    if( min < max )
    {
        const int length = (max - min);
        return rand()%length + min;
    }
    else
    {
        assert(!"Not support!");
        return 0;
    }
}

int GetDigit1(const int num)
{
    return num%10;
}

int GetDigit2(const int num)
{
    return (num/10)%10;
}

int GetDigit3(const int num)
{
    return (num/100)%10;
}

int Boolean(const bool cond)
{
    return (cond)?1:0;
}
#endif

// NEAT
/*
x
if x%2 == 0 return 1
else return 0
*/
int GetCorrectAns(const float num)
{
    return ((int)(num)%2) == 0;
}

float GetRandomWeight()
{
    const float ret = GetRandomFloat(-1, 1, 0.001f);
    assert(ret >= -1.0 && ret <= 1.0);
    return ret;
}

int AiInitWeight(WeightTable * weight_table)
{
    // init input level
    for(int i=0; i<INPUT_LEVEL_WEIGHT_INDEX_MAX; i++)
    {
        weight_table->input_level_weight[i] = GetRandomWeight();
    }
    // init proc level
    for(int level=0; level<WEIGHT_LEVEL_MAX; level++)
    {
        for(int i=0; i<WEIGHT_INDEX_MAX; i++)
        {
            weight_table->proc_level_weight[level][i] = GetRandomWeight();
        }
    }
    // init ans level
    for(int i=0; i<ANS_LEVEL_WEIGHT_INDEX_MAX; i++)
    {
        weight_table->ans_level_weight[i] = GetRandomWeight();
    }
    return 0;
}

int AiInitWeightAll(WeightTable weight_table_array[/*TRAIN_AI_NUM*/], const int array_size)
{
    for( int i=0; i<array_size; i++)
    {
        AiInitWeight(&weight_table_array[i]);
    }
    return 0;
}

int AiPrintWeight(const WeightTable * const weight)
{
    float num;
    // input level
    printf("input:\n");
    for(int i=0; i<INPUT_LEVEL_WEIGHT_INDEX_MAX; i++)
    {
        num = weight->input_level_weight[i];
        printf("%+6.3f, ",num);
    }
    // proc level
    printf("weight:\n");
    for(int level=0; level<WEIGHT_LEVEL_MAX; level++)
    {
        for(int i=0; i<WEIGHT_INDEX_MAX; i++)
        {
            num = weight->proc_level_weight[level][i];
            printf("%+6.3f, ",num);
        }
        printf("\n");
    }
    // ans level
    printf("ans:\n");
    for(int i=0; i<ANS_LEVEL_WEIGHT_INDEX_MAX; i++)
    {
        num = weight->ans_level_weight[i];
        printf("%+6.3f, ",num);
    }

    return 0;
}

float MutateWeight(const float num, const float range)
{
    float ret = num;
    ret = ret + GetRandomFloat(-1.f * range, range, 0.001f );
    return ret;
}

int AiMutate(WeightTable * weight_table, const float range)
{
    for(int i=0; i<WEIGHT_ALL_INDEX; i++)
    {
        weight_table->all_weight[i] = MutateWeight(weight_table->all_weight[i], range);
    }
    return 0;
}

int AiVaria(WeightTable * weight_table, const WeightTable * const target1, const WeightTable * const target2, const float range)
{
    for(int i=0; i<WEIGHT_ALL_INDEX; i++)
    {
        weight_table->all_weight[i] = (target2->all_weight[i] + target1->all_weight[i]) * .5f;
            MutateWeight(weight_table->all_weight[i], range);
    }
    if( range != 0 )
    {
        AiMutate(weight_table, range);
    }
    return 0;
}

int AiExec(const float num, const WeightTable * const weight_table)
{
    CellTable cell_table;

#if 1
    // input level
    // digit to cell[0]
    const int num_int = (int)num;
    cell_table.input_level_cell[ 0] = (float)Boolean(GetDigit1(num_int) == 0);
    cell_table.input_level_cell[ 1] = (float)Boolean(GetDigit1(num_int) == 1);
    cell_table.input_level_cell[ 2] = (float)Boolean(GetDigit1(num_int) == 2);
    cell_table.input_level_cell[ 3] = (float)Boolean(GetDigit1(num_int) == 3);
    cell_table.input_level_cell[ 4] = (float)Boolean(GetDigit1(num_int) == 4);
    cell_table.input_level_cell[ 5] = (float)Boolean(GetDigit1(num_int) == 5);
    cell_table.input_level_cell[ 6] = (float)Boolean(GetDigit1(num_int) == 6);
    cell_table.input_level_cell[ 7] = (float)Boolean(GetDigit1(num_int) == 7);
    cell_table.input_level_cell[ 8] = (float)Boolean(GetDigit1(num_int) == 8);
    cell_table.input_level_cell[ 9] = (float)Boolean(GetDigit1(num_int) == 9);

    cell_table.input_level_cell[10] = (float)Boolean(GetDigit2(num_int) == 0);
    cell_table.input_level_cell[11] = (float)Boolean(GetDigit2(num_int) == 1);
    cell_table.input_level_cell[12] = (float)Boolean(GetDigit2(num_int) == 2);
    cell_table.input_level_cell[13] = (float)Boolean(GetDigit2(num_int) == 3);
    cell_table.input_level_cell[14] = (float)Boolean(GetDigit2(num_int) == 4);
    cell_table.input_level_cell[15] = (float)Boolean(GetDigit2(num_int) == 5);
    cell_table.input_level_cell[16] = (float)Boolean(GetDigit2(num_int) == 6);
    cell_table.input_level_cell[17] = (float)Boolean(GetDigit2(num_int) == 7);
    cell_table.input_level_cell[18] = (float)Boolean(GetDigit2(num_int) == 8);
    cell_table.input_level_cell[19] = (float)Boolean(GetDigit2(num_int) == 9);

    cell_table.input_level_cell[20] = (float)Boolean(GetDigit3(num_int) == 0);
    cell_table.input_level_cell[21] = (float)Boolean(GetDigit3(num_int) == 1);
    cell_table.input_level_cell[22] = (float)Boolean(GetDigit3(num_int) == 2);
    cell_table.input_level_cell[23] = (float)Boolean(GetDigit3(num_int) == 3);
    cell_table.input_level_cell[24] = (float)Boolean(GetDigit3(num_int) == 4);
    cell_table.input_level_cell[25] = (float)Boolean(GetDigit3(num_int) == 5);
    cell_table.input_level_cell[26] = (float)Boolean(GetDigit3(num_int) == 6);
    cell_table.input_level_cell[27] = (float)Boolean(GetDigit3(num_int) == 7);
    cell_table.input_level_cell[28] = (float)Boolean(GetDigit3(num_int) == 8);
    cell_table.input_level_cell[29] = (float)Boolean(GetDigit3(num_int) == 9);

    // input level to proc level 1
    {
        int cell_index=0,weight_index=0;
        for(int i=0; i<INPUT_LEVEL_CELL_INDEX_MAX-1; i++)
        {
            for(int j=i+1; j<INPUT_LEVEL_CELL_INDEX_MAX; j++)
            {
                cell_table.proc_cell[0][cell_index++] = cell_table.input_level_cell[i] * weight_table->input_level_weight[weight_index++] +
                                                        cell_table.input_level_cell[j] * weight_table->input_level_weight[weight_index++];
            }
        }
        assert(cell_index == CELL_INDEX_MAX);
        assert(weight_index == INPUT_LEVEL_WEIGHT_INDEX_MAX);
    }

    // proc level
    for(int level=1; level<CELL_LEVEL_MAX; level++)
    {
        int weight_index = 0;
        const int last_level = level-1;
        for(int cell_index=0; cell_index<CELL_INDEX_MAX; cell_index++)
        {
            float sum = 0.f;
            for(int cell_last_index=0; cell_last_index<CELL_INDEX_MAX; cell_last_index++)
            {
                sum += cell_table.proc_cell[last_level][cell_last_index] * weight_table->proc_level_weight[last_level][weight_index++];
            }
            cell_table.proc_cell[level][cell_index] = sum;
            //= sum > 0 ? 1.f : 0.f;
        }
        assert(weight_index == WEIGHT_INDEX_MAX);
    }

    // ans level
    for(int cell_ans_level_index=0; cell_ans_level_index<ANS_LEVEL_CELL_INDEX_MAX; cell_ans_level_index++)
    {
        float sum = 0.f;
        int weight_ans_level_index = 0;
        const int last_level = CELL_LEVEL_MAX-1;
        for(int cell_last_index=0; cell_last_index<CELL_INDEX_MAX; cell_last_index++)
        {
            sum += cell_table.proc_cell[last_level][cell_last_index] * weight_table->proc_level_weight[last_level][weight_ans_level_index++];
        }
        assert(weight_ans_level_index == ANS_LEVEL_WEIGHT_INDEX_MAX);
        cell_table.ans_level_cell[cell_ans_level_index] = sum > 0 ? 1.f : 0.f;
    }
#endif


#if /* print */ 0
    float ansArr[SIZE] = {0};
    float ans2 = 0;
    for( int j=0; j<SIZE; j++ )
    {
        float temp = 0;
        for( int i=0; i<SIZE; i++ )
        {
            temp += cell[LAST_LEVEL][i] * weight->proc_level_weight[LAST_LEVEL][i][j];
        }
        ansArr[j] = temp > 0 ? 1.f : 0.f;
    }
    int ans = ((int)ansArr[0] << 1) + (int)ansArr[1];

    // print
    if( false )
    {
        printf("%-8.2f", num);
        for( int level=0; level<LEVEL; level++ )
        {
            printf("[%6.2f] ", cell[level][0]);
        }
        printf("%.2f\n", ans);

        for( int i=1; i<SIZE; i++ )
        {
            printf("\t");
            for( int level=0; level<LEVEL; level++ )
            {
                printf("[%6.2f] ", cell[level][i]);
            }
            printf("\n");
        }
    }
#endif

    // return
    return (int)cell_table.ans_level_cell[0];
}


#if /* FILE */ 1
#define FILENAME_ALL    "weight_all.ai"

int FileWrite(char * file_name, WeightTable * weight_table)
{
    FILE * fp = fopen(file_name, "wb" );
    fwrite(weight_table->all_weight, sizeof(WeightTable), 1, fp);
    return fclose(fp);
}

int FileWriteAll(WeightTable weight_table_array[/*TRAIN_AI_NUM*/], const int array_size)
{
    FILE * fp = fopen(FILENAME_ALL, "wb" );
    for(int i=0; i<array_size; i++)
    {
        fwrite((weight_table_array+i)->all_weight, sizeof(WeightTable), 1, fp);
    }
    return fclose(fp);
}

int FileRead(char * file_name, WeightTable * weight_table)
{
    FILE * fp = fopen(file_name, "rb" );
    fread(weight_table->all_weight, sizeof(WeightTable), 1, fp);
    return fclose(fp);
}

int FileReadAll(WeightTable weight_table_array[/*TRAIN_AI_NUM*/], const int array_size)
{
    FILE * fp = fopen(FILENAME_ALL, "rb" );
    for(int i=0; i<array_size; i++)
    {
        fread((weight_table_array+i)->all_weight, sizeof(WeightTable), 1, fp);
    }
    return fclose(fp);
}

int InitWeightAndWriteToFileThenExit(WeightTable weight_table_array[], const int lock_seed, const int print_init, const int print_read, const int do_pause, const int do_exit)
{
    if( lock_seed )
    {
        srand(0);
    }

    AiInitWeightAll(weight_table_array,TRAIN_AI_NUM);
    if( print_init )
    for(int i=0; i<TRAIN_AI_NUM; i++)
    {
        printf("Init %d:\n", i);
        AiPrintWeight(&weight_table_array[i]);
    }

    FileWriteAll(weight_table_array, TRAIN_AI_NUM);

    memset( weight_table_array, 0, sizeof(weight_table_array[0])*TRAIN_AI_NUM);

    FileReadAll(weight_table_array, TRAIN_AI_NUM);

    if( print_read )
    for(int i=0; i<TRAIN_AI_NUM; i++)
    {
        printf("Read %d:\n", i);
        AiPrintWeight(&weight_table_array[i]);
    }

    printf("save!");
    do_pause && getchar();
    if( do_exit )
    {
        exit(0);
    }
    return 0;
}

int TryReadWeightFile(WeightTable weight_table_array[/*TRAIN_AI_NUM*/], const int lock_seed, const int print_init, const int print_read, const int do_pause, const int do_exit)
{
    FILE * fp = fopen(FILENAME_ALL, "r" );
    if( fp == NULL )
    {
        InitWeightAndWriteToFileThenExit( weight_table_array, lock_seed, print_init, print_read, do_pause, do_exit );
    }
    else
    {
        fclose(fp);
    }

    return FileReadAll(weight_table_array, TRAIN_AI_NUM);
}

#endif


float RepeatTrainAi(WeightTable * weight_table, const int repeat_time, const int lock_seed, const int lock_num, const int print_per_ans, const int do_per_ans_pause)
{
    float reward;
    float num;

    if( lock_seed )
    {
        srand(0);
    }
    reward = 0;

    for(int cnt=0;cnt<repeat_time;cnt++)
    {
        if( lock_num )
        {
            num = (float)cnt;
        }
        else
        {
            num = (float)(rand()%1000);
        }

        int ai_ans;
        int correct_ans;
        correct_ans = GetCorrectAns(num);
        ai_ans = AiExec(num,weight_table);

        if( correct_ans == ai_ans )
        {
            reward += 1;
        }

        print_per_ans && printf("#%-3d  %-2.f -> %d %d (%.f) [%s]\n", cnt, num, correct_ans, ai_ans, reward, correct_ans == ai_ans ? "Yes" : "No");
        do_per_ans_pause && getchar();
    }
    return reward;
}

float TrainAiEnter(WeightTable weight_table_array[], float reward_array[], const int repeat_time, const int lock_seed, const int lock_num, const int max_ai_train_index, const int print_weight, const int print_per_ans, const int do_per_index_pause, const int do_per_ans_pause)
{
    float max_reward = 0;
    WeightTable * weight_table;

    for(int ai_train_index=0;ai_train_index<max_ai_train_index;ai_train_index++)
    {
        weight_table = &weight_table_array[ai_train_index];
        print_weight && AiPrintWeight(weight_table);

        float reward;
        reward = RepeatTrainAi(weight_table, repeat_time, lock_seed, lock_num, print_per_ans, do_per_ans_pause);
        if( reward > max_reward )
        {
            max_reward = reward;
        }
        reward_array[ai_train_index] = reward;
        do_per_index_pause && getchar();
    }
    return max_reward;
}

int SortAiWeightIndex(int need_sort_index_array[], float reward_array[], const int max_index)
{
    int index = 0;
    float max_reward = reward_array[0];
    for(int i=1; i<max_index; i++)
    {
        if( reward_array[i] > max_reward )
        {
            max_reward = reward_array[i];
        }
    }

    for(;;)
    {
        for(int i=0; i<max_index; i++)
        {
            if( reward_array[i] == max_reward )
            {
                need_sort_index_array[index] = i;
                index++;
            }
        }
        const float max_last_reward = max_reward;
        int exit = true;
        max_reward = INT32_MIN;
        for(int i=0; i<max_index; i++)
        {
            if( reward_array[i] < max_last_reward && reward_array[i] > max_reward )
            {
                max_reward = reward_array[i];
                exit = false;
            }
        }
        if( exit )
        {
            break;
        }
    }

    assert(index == max_index);
    return index;
}

int UpdateAiWeight(WeightTable weight_table_array[], int sorted_index_array[], const int max_index)
{
    const float range = 0.5f;
    //WeightTable weight_table_trained_array[KEEP_TRAINED_AI];
    WeightTable * weight_table_trained_array;
    weight_table_trained_array = (WeightTable*)malloc(sizeof(WeightTable) * KEEP_TRAINED_AI);

    // Keep weight and mutate
    const int keep_index = KEEP_TRAINED_AI;
    for(int i=0; i<keep_index; i++)
    {
        const int index = sorted_index_array[i];
        memcpy(&weight_table_trained_array[i], &weight_table_array[index], sizeof(weight_table_array[0]));
    }
    for(int i=0; i<keep_index; i++)
    {
        memcpy(&weight_table_array[i], &weight_table_trained_array[i], sizeof(weight_table_array[0]));
    }

    int index = KEEP_TRAINED_AI;
    // Do Mutate
    for(int i=0; i<keep_index; i++,index++)
    {
        memcpy(&weight_table_array[index], &weight_table_trained_array[i], sizeof(weight_table_array[0]));
        AiMutate(&weight_table_array[index], range);
    }

    // Do Varia
    for(int i=0; i<keep_index; i++)
    {
        for(int j=i+1; j<keep_index; j++)
        {
            AiVaria(&weight_table_array[index], &weight_table_trained_array[i], &weight_table_trained_array[j], range);
            index++;
        }
    }

    // Init new weight
    for(int i=index; i<max_index; i++)
    {
        AiInitWeight(&weight_table_array[i]);
    }

    free(weight_table_trained_array);

    return 0;
}


int main()
{
    srand((unsigned int)time(NULL));

    // Can not use local var because of 0x7fffffff
    WeightTable * g_weight_table_array;
    g_weight_table_array = (WeightTable*)malloc(sizeof(WeightTable) * TRAIN_AI_NUM);

    // printf("%d\n", sizeof(WeightTable));
    // printf("%d\n", sizeof(g_weight_table_array));
    // printf("%d\n", CELL_LEVEL_MAX);
    // printf("%d\n", 0x7fffffff);
    // return 0;

    assert(CELL_LEVEL_MAX == (WEIGHT_LEVEL_MAX + 1));

    float reward_array[TRAIN_AI_NUM];
    int sorted_index_array[TRAIN_AI_NUM];
    float max_reward;

    TryReadWeightFile( g_weight_table_array, false, false, false, true, false );

    int repeat_time, lock_seed, lock_num, max_ai_train_index, print_weight, print_per_ans, do_per_index_pause, do_per_ans_pause;
#if 0
    {
        // Print 1
        repeat_time         = 1000;
        lock_seed           = true;
        lock_num            = true;
        max_ai_train_index  = 1;
        print_weight        = false;
        print_per_ans       = true;
        do_per_index_pause  = true;
        do_per_ans_pause    = true;
    }
#elif 0
    {
        // Check
        repeat_time         = 1000;
        lock_seed           = true;
        lock_num            = true;
        max_ai_train_index  = TRAIN_AI_NUM;
        print_weight        = false;
        print_per_ans       = false;
        do_per_index_pause  = false;
        do_per_ans_pause    = false;
    }
#elif 1
    {
        // train
        repeat_time         = 500;
        lock_seed           = true;
        lock_num            = false;
        max_ai_train_index  = TRAIN_AI_NUM;
        print_weight        = false;
        print_per_ans       = false;
        do_per_index_pause  = false;
        do_per_ans_pause    = false;
    }
#endif

    for(int train_cnt=0;;train_cnt++)
    {
        const DWORD tick_start = GetTickCount();
        max_reward = TrainAiEnter(g_weight_table_array, reward_array, repeat_time, lock_seed, lock_num, max_ai_train_index, print_weight, print_per_ans, do_per_index_pause, do_per_ans_pause);
        const DWORD tick_end = GetTickCount();
        printf("tick: %lu\n", tick_end - tick_start);

        SortAiWeightIndex(sorted_index_array, reward_array, max_ai_train_index);

        // if win, save and exit
        if( max_reward >= repeat_time * 0.99999)
        {
            const int index = sorted_index_array[0];
            memcpy(&g_weight_table_array[0], &g_weight_table_array[index], sizeof(g_weight_table_array[0]));
            break;
        }

        // update AI
        UpdateAiWeight(g_weight_table_array, sorted_index_array, max_ai_train_index);


        float totalReward = 0;
        for(int i=0; i<max_ai_train_index; i++)
        {
            totalReward += reward_array[i];
        }

        printf("#%d (max %.f) (ave %.f)\n", train_cnt, max_reward, totalReward/max_ai_train_index);
    }

    printf("Perfect! (%.2f)\n", max_reward);
    FileWriteAll(g_weight_table_array, TRAIN_AI_NUM);
    getchar();

    return 0;
}
