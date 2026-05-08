#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1d ATR(14) < median ATR(14) of last 50 days AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period Donchian low AND 1d ATR(14) < median ATR(14) of last 50 days AND volume > 1.5x 20-period average.
# Exit when price crosses back below Donchian high (for long) or above Donchian low (for short).
# Donchian channels provide clear breakout levels. Low volatility filter avoids false breakouts in choppy markets.
# Volume surge confirms institutional participation. Target: 100-150 total trades over 4 years (25-38/year).

name = "4h_Donchian_20_1dATR14_LowVol_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period TR
    tr2[0] = 0  # No previous close for first period
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate median ATR over last 50 days for volatility regime filter
    atr_median_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    low_vol_filter = atr_14 < atr_median_50  # Only trade when volatility is below median
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    low_vol_filter_aligned = align_htf_to_ltf(prices, df_1d, low_vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for ATR median
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_median_50_aligned[i]) or 
            np.isnan(low_vol_filter_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high, low volatility, volume surge
            long_cond = (close[i] > donchian_high[i]) and low_vol_filter_aligned[i] and volume_filter[i]
            # Short conditions: break below Donchian low, low volatility, volume surge
            short_cond = (close[i] < donchian_low[i]) and low_vol_filter_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross back below Donchian high
            if close[i] < donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross back above Donchian low
            if close[i] > donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals