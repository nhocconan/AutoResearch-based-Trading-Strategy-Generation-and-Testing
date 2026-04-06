#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND weekly trend up (price > weekly SMA50) AND volume > 1.5x avg
# Short when price breaks below Donchian(20) low AND weekly trend down (price < weekly SMA50) AND volume > 1.5x avg
# Exit when price crosses midline (10-period average) or opposite breakout occurs
# Uses 6h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter

name = "6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
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
    
    # Donchian Channel (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Midline for exit (10-period average of high/low)
    midline = ((pd.Series(high).rolling(window=10, min_periods=10).mean() + 
                pd.Series(low).rolling(window=10, min_periods=10).mean()) / 2).values
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(midline[i]) or \
           np.isnan(weekly_sma_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= midline[i] or close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= midline[i] or close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian high AND weekly trend up (price > weekly SMA) AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_sma_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly trend down (price < weekly SMA) AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_sma_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1w trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND weekly trend up (price > weekly SMA50) AND volume > 1.5x avg
# Short when price breaks below Donchian(20) low AND weekly trend down (price < weekly SMA50) AND volume > 1.5x avg
# Exit when price crosses midline (10-period average) or opposite breakout occurs
# Uses 6h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter

name = "6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
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
    
    # Donchian Channel (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # Midline for exit (10-period average of high/low)
    midline = ((pd.Series(high).rolling(window=10, min_periods=10).mean() + 
                pd.Series(low).rolling(window=10, min_periods=10).mean()) / 2).values
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_sma = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(midline[i]) or \
           np.isnan(weekly_sma_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= midline[i] or close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= midline[i] or close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian high AND weekly trend up (price > weekly SMA) AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > weekly_sma_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly trend down (price < weekly SMA) AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < weekly_sma_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals