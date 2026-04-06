#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation
# Long when price breaks above 12h Donchian high AND 1d EMA(50) up AND volume > 1.5x avg
# Short when price breaks below 12h Donchian low AND 1d EMA(50) down AND volume > 1.5x avg
# Exit when price crosses 12h midline OR volume dries up
# Designed for 12h timeframe to reduce trade frequency (target: 50-150 total trades over 4 years)
# Works in both bull and bear markets by following trend with volume confirmation

name = "12h_donchian_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for all indicators to be ready
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian high AND EMA up AND volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA down AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation
# Long when price breaks above 12h Donchian high AND 1d EMA(50) up AND volume > 1.5x avg
# Short when price breaks below 12h Donchian low AND 1d EMA(50) down AND volume > 1.5x avg
# Exit when price crosses 12h midline OR volume dries up
# Designed for 12h timeframe to reduce trade frequency (target: 50-150 total trades over 4 years)
# Works in both bull and bear markets by following trend with volume confirmation

name = "12h_donchian_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for all indicators to be ready
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume confirmation
            # Long: price breaks above Donchian high AND EMA up AND volume confirmation
            if (close[i] > donchian_high[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA down AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals