#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter (HMA21) + volume confirmation
# Long when price breaks above Donchian upper band AND price > HMA(21) on weekly
# Short when price breaks below Donchian lower band AND price < HMA(21) on weekly
# Exit when price crosses opposite Donchian band or HMA crosses in opposite direction
# Target: 75-200 total trades over 4 years (19-50/year) for 1d timeframe
# Works in bull markets via breakouts, bear markets via short breakdowns

name = "1d_donchian20_hma21_vol_v1"
timeframe = "1d"
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
    
    # Donchian Channel (20-period) - price channel breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    
    # Weekly HMA (21-period) - trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n))
    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights, 'valid') / weights.sum()
    
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2n_minus_n = 2 * wma_half - wma_full
        return wma(wma_2n_minus_n, sqrt)
    
    # Calculate HMA for weekly data
    hma_weekly = hma(weekly_close, 21)
    hma_weekly_aligned = align_htf_to_ltf(prices, df_1w, hma_weekly)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(hma_weekly_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < donchian_lower[i] or close[i] < hma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_upper[i] or close[i] > hma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout + HMA trend + volume confirmation
            # Long: price breaks above Donchian upper AND price > weekly HMA
            if (close[i] > donchian_upper[i] and close[i] > hma_weekly_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly HMA
            elif (close[i] < donchian_lower[i] and close[i] < hma_weekly_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter (HMA21) + volume confirmation
# Long when price breaks above Donchian upper band AND price > HMA(21) on weekly
# Short when price breaks below Donchian lower band AND price < HMA(21) on weekly
# Exit when price crosses opposite Donchian band or HMA crosses in opposite direction
# Target: 75-200 total trades over 4 years (19-50/year) for 1d timeframe
# Works in bull markets via breakouts, bear markets via short breakdowns

name = "1d_donchian20_hma21_vol_v1"
timeframe = "1d"
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
    
    # Donchian Channel (20-period) - price channel breakout
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    
    # Weekly HMA (21-period) - trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n))
    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights, 'valid') / weights.sum()
    
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2n_minus_n = 2 * wma_half - wma_full
        return wma(wma_2n_minus_n, sqrt)
    
    # Calculate HMA for weekly data
    hma_weekly = hma(weekly_close, 21)
    hma_weekly_aligned = align_htf_to_ltf(prices, df_1w, hma_weekly)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(hma_weekly_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < donchian_lower[i] or close[i] < hma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_upper[i] or close[i] > hma_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout + HMA trend + volume confirmation
            # Long: price breaks above Donchian upper AND price > weekly HMA
            if (close[i] > donchian_upper[i] and close[i] > hma_weekly_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly HMA
            elif (close[i] < donchian_lower[i] and close[i] < hma_weekly_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>