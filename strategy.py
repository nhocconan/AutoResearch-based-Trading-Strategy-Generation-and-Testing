#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 24-period high OR volume confirmation fails OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] >= ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            # Long: price breaks above 24-period high AND volume > 1.5x average AND daily close above EMA
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                close_1d[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 24-period low AND volume > 1.5x average AND daily close below EMA
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg_12h_aligned[i] and 
                  close_1d[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and daily trend filter
# Long when price breaks above 24-period high (4d equivalent) AND 12h volume > 1.5x 20-period average AND daily close above 20-day EMA
# Short when price breaks below 24-period low AND 12h volume > 1.5x average AND daily close below 20-day EMA
# Uses volume to confirm breakouts and daily trend to filter direction.
# Target: 75-200 total trades over 4 years (19-50/year) for optimal frequency.

name = "6h_donchian24_12h_vol_1d_ema"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (24-period for 6h = 4 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=24, min_periods=24).max()
    donchian_low = low_series.rolling(window=24, min_periods=24).min()
    
    # 12h volume average
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_avg_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    volume_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_avg_12h)
    
    # Daily EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if data not available
        if np.isnan(volume_avg_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or conditions fail
        if position == 1:  # long position
            # Exit: price breaks below 24-period low OR volume confirmation fails OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                volume[i] <= volume_avg_12h_aligned[i] or 
                close_1d[i] <= ema_1d_aligned[i]):
                signals[i] = 0.0