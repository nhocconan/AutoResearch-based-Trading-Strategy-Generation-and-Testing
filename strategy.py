#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Bollinger Band breakout with daily volume confirmation
# Long when price breaks above upper BB AND daily volume > 20-period average
# Short when price breaks below lower BB AND daily volume > 20-period average
# Exit when price returns to middle BB (mean reversion)
# Bollinger Bands capture volatility expansion/contraction, volume confirms breakout strength
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing volatility bursts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands on 12h: 20-period SMA, 2 std dev
    close_12h = df_12h['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20
    
    # Calculate daily volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    
    # Align indicators to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb.values)
    middle_bb_aligned = align_htf_to_ltf(prices, df_12h, middle_bb.values)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        current_volume = volume[i]
        
        if position == 0:
            # Long setup: price breaks above upper BB AND volume above average
            if (price > upper_bb_aligned[i] and 
                current_volume > vol_avg_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below lower BB AND volume above average
            elif (price < lower_bb_aligned[i] and 
                  current_volume > vol_avg_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB
            if price < middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle BB
            if price > middle_bb_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_BollingerBreakout_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0