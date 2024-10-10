
#include "VP.h"

#include "../utils/analysis_board/vp_analysis_board.h"

#if _1_1_1_sample

int main() {
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    board.display();
}

#endif
