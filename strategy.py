#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above Donchian(20) high AND daily ATR(14) < median ATR(100) (low volatility) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND daily ATR(14) < median ATR(100) AND volume > 1.5x average
# Exit when price crosses the Donchian midline (10-period) in opposite direction
# Designed to capture breakouts in low volatility environments with institutional participation.
# Target: 60-140 total trades over 4 years (15-35/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate ATR on 1d (14-period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Calculate median ATR on 1d (100-period) for volatility regime filter
    atr_median = pd.Series(atr_14).rolling(window=100, min_periods=100).median()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ATR values aligned to 4h timeframe
        atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14.values)
        atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median.values)
        atr_val = atr_14_aligned[i]
        atr_med_val = atr_median_aligned[i]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above Donchian high AND low volatility regime AND volume confirmation
            if (high_val > highest_high[i] and atr_val < atr_med_val and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low AND low volatility regime AND volume confirmation
            elif (low_val < lowest_low[i] and atr_val < atr_med_val and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close_val < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close_val > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ATR_Volume"
timeframe = "4h"
leverage = 1.0