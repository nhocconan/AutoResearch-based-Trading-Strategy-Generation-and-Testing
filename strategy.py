#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 12h EMA filter.
# Camarilla levels (R1/S1, R2/S2) from 12h data provide intraday support/resistance.
# Breakouts above R1 or below S1 with volume confirmation signal momentum.
# 12h EMA34 filters trades to align with higher timeframe trend.
# Works in both bull and breakouts (buying strength) and bear markets (selling weakness).
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag.
name = "6h_Camarilla_R1_S1_Breakout_Volume_12hEMA34"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # Previous day's range for Camarilla calculation
    prev_high = high_12h.shift(1)
    prev_low = low_12h.shift(1)
    prev_close = close_12h.shift(1)
    
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    r2 = prev_close + range_hl * 1.1 / 6
    s2 = prev_close - range_hl * 1.1 / 6
    
    # Calculate EMA34 on 12h close
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 12h indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2.values)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: current volume > 1.8 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume and above EMA
            if close_val > r1_val and volume_spike[i] and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and below EMA
            elif close_val < s1_val and volume_spike[i] and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA or reach S2 (support)
            if close_val < ema_val or close_val < s2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA or reach R2 (resistance)
            if close_val > ema_val or close_val > r2_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals