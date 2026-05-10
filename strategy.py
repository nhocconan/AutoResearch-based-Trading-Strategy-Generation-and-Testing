#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses 12h timeframe with 1-day trend (EMA34) and volume confirmation to filter breakouts.
# Enters long when price breaks above daily R1 in daily uptrend (close > EMA34) with volume > 2x 20-period average.
# Enters short when price breaks below daily S1 in daily downtrend (close < EMA34) with volume confirmation.
# Exits when price returns to opposite level (S1 for long, R1 for short) or trend reverses.
# Uses 1-day EMA34 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25.

name = "12H_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend (EMA34) and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r1_level = prev_close + 1.1 * (pivot_range / 12)  # R1 = C + 1.1*(H-L)/12
    s1_level = prev_close - 1.1 * (pivot_range / 12)  # S1 = C - 1.1*(H-L)/12
    
    # Align pivot levels to 12h timeframe (available after 1d bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Volume filter: volume > 2x 20-period average on 12h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 in daily uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 in daily downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to S1 or trend reverses to downtrend
            if (close[i] < s1_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to R1 or trend reverses to uptrend
            if (close[i] > r1_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals