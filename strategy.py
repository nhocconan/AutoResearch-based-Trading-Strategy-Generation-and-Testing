#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12hma50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12h HMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    half_length = 50 // 2
    sqrt_length = int(np.sqrt(50))
    wma_half = pd.Series(close_12h).rolling(window=half_length, min_periods=half_length).mean().values
    wma_full = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_50 = wma_sqrt
    hma_50_aligned = align_htf_to_ltf(prices, df_12h, hma_50)
    
    # Volume confirmation: volume > 1.6x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.6 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(hma_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > hma_50_aligned[i]:
                    # Bullish breakout above Donchian high with 12h uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < hma_50_aligned[i]:
                    # Bearish breakdown below Donchian low with 12h downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12hma50_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12h HMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    half_length = 50 // 2
    sqrt_length = int(np.sqrt(50))
    wma_half = pd.Series(close_12h).rolling(window=half_length, min_periods=half_length).mean().values
    wma_full = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    wma_sqrt = pd.Series(2 * wma_half - wma_full).rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_50 = wma_sqrt
    hma_50_aligned = align_htf_to_ltf(prices, df_12h, hma_50)
    
    # Volume confirmation: volume > 1.6x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.6 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(hma_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > hma_50_aligned[i]:
                    # Bullish breakout above Donchian high with 12h uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < hma_50_aligned[i]:
                    # Bearish breakdown below Donchian low with 12h downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals