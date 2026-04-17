#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d Volume Spike and 1w EMA200 Trend Filter.
Long when Williams %R crosses above -80 (oversold recovery) AND 1d volume > 1.5x 20-period average (strong participation) AND price > 1w EMA200 (bullish long-term trend).
Short when Williams %R crosses below -20 (overbought rejection) AND 1d volume > 1.5x 20-period average AND price < 1w EMA200 (bearish long-term trend).
Exit when Williams %R returns to opposite extreme (> -20 for longs, < -80 for shorts) or weekly trend reverses.
Uses 1d for volume confirmation and Williams %R calculation, 1w for EMA200 trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion extremes, 
volume spike ensures institutional participation, weekly EMA200 filters for higher-timeframe trend alignment 
to avoid counter-trend traps in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Calculate 1d volume spike confirmation (current volume > 1.5x 20-period average)
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * volume_ma_20)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Align 1w EMA200 to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # boolean
        price = close[i]
        ema200 = ema200_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND volume spike AND price > 1w EMA200
            if i > start_idx and wr > -80 and williams_r_aligned[i-1] <= -80 and vol_spike and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND volume spike AND price < 1w EMA200
            elif i > start_idx and wr < -20 and williams_r_aligned[i-1] >= -20 and vol_spike and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -20 (overbought) OR price < 1w EMA200 (trend reversal)
            if wr > -20 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -80 (oversold) OR price > 1w EMA200 (trend reversal)
            if wr < -80 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_VolumeSpike_1wEMA200_Trend"
timeframe = "6h"
leverage = 1.0