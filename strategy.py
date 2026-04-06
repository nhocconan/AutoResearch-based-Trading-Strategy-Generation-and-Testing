#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 1.5x avg
# Exit when: opposite Donchian breakout occurs or price crosses 1d EMA(50) against position
# Uses daily trend to filter breakouts in strong moves, targeting 75-200 trades over 4 years

name = "4h_donchian20_1dema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price < 1d EMA(50)
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price > 1d EMA(50)
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donch_high[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high and above daily EMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below Donchian low and below daily EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 1.5x avg
# Exit when: opposite Donchian breakout occurs or price crosses 1d EMA(50) against position
# Uses daily trend to filter breakouts in strong moves, targeting 75-200 trades over 4 years

name = "4h_donchian20_1dema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR price < 1d EMA(50)
            if close[i] < donch_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR price > 1d EMA(50)
            if close[i] > donch_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donch_high[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high and above daily EMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_low[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below Donchian low and below daily EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals