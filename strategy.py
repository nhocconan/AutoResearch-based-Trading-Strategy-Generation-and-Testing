#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(15) trend filter and volume confirmation.
# Enter long when price breaks above daily Donchian(20) upper band in uptrend (weekly EMA15 rising).
# Enter short when price breaks below daily Donchian(20) lower band in downtrend (weekly EMA15 falling).
# Volume > 1.8x 20-day average confirms breakout strength.
# Exit on opposite daily Donchian breakout or when price crosses weekly EMA15.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Uses weekly trend to capture multi-month trends while avoiding whipsaws in sideways markets.
# Weekly EMA15 provides smoother trend signal than daily, reducing false breakouts.

name = "1d_donchian20_weeklyema15_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(15) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_15 = pd.Series(close_1w).ewm(span=15, adjust=False).mean().values
    ema_15_aligned = align_htf_to_ltf(prices, df_1w, ema_15)
    
    # Daily Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_15_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below daily Donchian lower OR crosses below weekly EMA15
            if close[i] < lowest_low_20[i] or close[i] < ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above daily Donchian upper OR crosses above weekly EMA15
            if close[i] > highest_high_20[i] or close[i] > ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA15 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_15_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_15_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(15) trend filter and volume confirmation.
# Enter long when price breaks above daily Donchian(20) upper band in uptrend (weekly EMA15 rising).
# Enter short when price breaks below daily Donchian(20) lower band in downtrend (weekly EMA15 falling).
# Volume > 1.8x 20-day average confirms breakout strength.
# Exit on opposite daily Donchian breakout or when price crosses weekly EMA15.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Uses weekly trend to capture multi-month trends while avoiding whipsaws in sideways markets.
# Weekly EMA15 provides smoother trend signal than daily, reducing false breakouts.

name = "1d_donchian20_weeklyema15_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(15) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_15 = pd.Series(close_1w).ewm(span=15, adjust=False).mean().values
    ema_15_aligned = align_htf_to_ltf(prices, df_1w, ema_15)
    
    # Daily Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_15_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below daily Donchian lower OR crosses below weekly EMA15
            if close[i] < lowest_low_20[i] or close[i] < ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above daily Donchian upper OR crosses above weekly EMA15
            if close[i] > highest_high_20[i] or close[i] > ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA15 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_15_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_15_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(15) trend filter and volume confirmation.
# Enter long when price breaks above daily Donchian(20) upper band in uptrend (weekly EMA15 rising).
# Enter short when price breaks below daily Donchian(20) lower band in downtrend (weekly EMA15 falling).
# Volume > 1.8x 20-day average confirms breakout strength.
# Exit on opposite daily Donchian breakout or when price crosses weekly EMA15.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Uses weekly trend to capture multi-month trends while avoiding whipsaws in sideways markets.
# Weekly EMA15 provides smoother trend signal than daily, reducing false breakouts.

name = "1d_donchian20_weeklyema15_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(15) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_15 = pd.Series(close_1w).ewm(span=15, adjust=False).mean().values
    ema_15_aligned = align_htf_to_ltf(prices, df_1w, ema_15)
    
    # Daily Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_15_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below daily Donchian lower OR crosses below weekly EMA15
            if close[i] < lowest_low_20[i] or close[i] < ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above daily Donchian upper OR crosses above weekly EMA15
            if close[i] > highest_high_20[i] or close[i] > ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA15 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_15_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_15_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(15) trend filter and volume confirmation.
# Enter long when price breaks above daily Donchian(20) upper band in uptrend (weekly EMA15 rising).
# Enter short when price breaks below daily Donchian(20) lower band in downtrend (weekly EMA15 falling).
# Volume > 1.8x 20-day average confirms breakout strength.
# Exit on opposite daily Donchian breakout or when price crosses weekly EMA15.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Uses weekly trend to capture multi-month trends while avoiding whipsaws in sideways markets.
# Weekly EMA15 provides smoother trend signal than daily, reducing false breakouts.

name = "1d_donchian20_weeklyema15_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(15) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_15 = pd.Series(close_1w).ewm(span=15, adjust=False).mean().values
    ema_15_aligned = align_htf_to_ltf(prices, df_1w, ema_15)
    
    # Daily Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_15_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below daily Donchian lower OR crosses below weekly EMA15
            if close[i] < lowest_low_20[i] or close[i] < ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above daily Donchian upper OR crosses above weekly EMA15
            if close[i] > highest_high_20[i] or close[i] > ema_15_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA15 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > highest_high_20[i] and close[i] > ema_15_aligned[i]:
                    # Breakout above upper band in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low_20[i] and close[i] < ema_15_aligned[i]:
                    # Breakdown below lower band in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals

--- End of file ---