#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h Choppiness Index regime filter + volume confirmation
# Long when price breaks above Donchian upper band AND Choppiness Index > 61.8 (ranging) AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND Choppiness Index > 61.8 AND volume > 1.5x average
# Exit when price reverts to Donchian middle (20-day SMA) OR Choppiness Index < 38.2 (trending)
# Uses 4h timeframe to capture major moves, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by fading extremes in ranging conditions and following breakouts in trending

name = "4h_donchian20_12h_chop_vol_v1"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((donchian_upper + donchian_lower) / 2)
    
    # Choppiness Index (14-period) from 12h timeframe for regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate TR for 12h
    tr_12h = []
    for i in range(len(high_12h)):
        if i == 0:
            tr = high_12h[i] - low_12h[i]
        else:
            tr = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        tr_12h.append(tr)
    
    tr_12h = np.array(tr_12h)
    tr_sum_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum()
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    range_12h = highest_high_12h - lowest_low_12h
    
    chop_12h = 100 * np.log10(tr_sum_12h / (range_12h + 1e-10)) / np.log10(14)
    chop_12h = chop_12h.values
    
    # Align 12h Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price reverts to middle OR market starts trending (chop < 38.2)
        if position == 1:  # long position
            if close[i] <= donchian_middle[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_middle[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout in ranging market with volume confirmation
            # Long: price breaks above upper band in ranging market + volume confirmation
            if (close[i] > donchian_upper[i] and chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band in ranging market + volume confirmation
            elif (close[i] < donchian_lower[i] and chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h Choppiness Index regime filter + volume confirmation
# Long when price breaks above Donchian upper band AND Choppiness Index > 61.8 (ranging) AND volume > 1.5x average
# Short when price breaks below Donchian lower band AND Choppiness Index > 61.8 AND volume > 1.5x average
# Exit when price reverts to Donchian middle (20-day SMA) OR Choppiness Index < 38.2 (trending)
# Uses 4h timeframe to capture major moves, targets 75-200 total trades over 4 years
# Works in both bull/bear markets by fading extremes in ranging conditions and following breakouts in trending

name = "4h_donchian20_12h_chop_vol_v1"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = ((donchian_upper + donchian_lower) / 2)
    
    # Choppiness Index (14-period) from 12h timeframe for regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate TR for 12h
    tr_12h = []
    for i in range(len(high_12h)):
        if i == 0:
            tr = high_12h[i] - low_12h[i]
        else:
            tr = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        tr_12h.append(tr)
    
    tr_12h = np.array(tr_12h)
    tr_sum_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum()
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    range_12h = highest_high_12h - lowest_low_12h
    
    chop_12h = 100 * np.log10(tr_sum_12h / (range_12h + 1e-10)) / np.log10(14)
    chop_12h = chop_12h.values
    
    # Align 12h Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(chop_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price reverts to middle OR market starts trending (chop < 38.2)
        if position == 1:  # long position
            if close[i] <= donchian_middle[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_middle[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout in ranging market with volume confirmation
            # Long: price breaks above upper band in ranging market + volume confirmation
            if (close[i] > donchian_upper[i] and chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band in ranging market + volume confirmation
            elif (close[i] < donchian_lower[i] and chop_aligned[i] > 61.8 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals