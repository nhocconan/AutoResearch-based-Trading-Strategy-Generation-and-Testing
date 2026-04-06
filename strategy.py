#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20-period) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
# Uses 4h timeframe to capture medium-term trends, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by following 1d trend direction and using volume confirmation

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_1d = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian upper band AND price > 1d EMA(50) AND volume > 1.5x average
            if (close[i] > highest_high[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA(50) AND volume > 1.5x average
            elif (close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20-period) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
# Uses 4h timeframe to capture medium-term trends, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by following 1d trend direction and using volume confirmation

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_1d = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian upper band AND price > 1d EMA(50) AND volume > 1.5x average
            if (close[i] > highest_high[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA(50) AND volume > 1.5x average
            elif (close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20-period) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
# Uses 4h timeframe to capture medium-term trends, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by following 1d trend direction and using volume confirmation

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_1d = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian upper band AND price > 1d EMA(50) AND volume > 1.5x average
            if (close[i] > highest_high[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA(50) AND volume > 1.5x average
            elif (close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian upper band (20-period) AND price > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian lower band (20-period) AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
# Uses 4h timeframe to capture medium-term trends, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by following 1d trend direction and using volume confirmation

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_1d = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back inside Donchian bands OR trend weakens (price crosses 1d EMA)
        if position == 1:  # long position
            if close[i] <= lowest_low[i] or close[i] <= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= highest_high[i] or close[i] >= ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian upper band AND price > 1d EMA(50) AND volume > 1.5x average
            if (close[i] > highest_high[i] and close[i] > ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND price < 1d EMA(50) AND volume > 1.5x average
            elif (close[i] < lowest_low[i] and close[i] < ema_1d_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

--- End of file ---