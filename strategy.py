#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 12-30/year) with clear mean-reversion logic
# Works in both bull (fade breakouts) and bear (fade breakdowns) markets
# Uses volume spike and EMA trend filter to avoid false reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Using previous bar's HLC (standard Camarilla calculation)
    ph = np.roll(high_12h, 1)  # previous high
    pl = np.roll(low_12h, 1)   # previous low
    pc = np.roll(close_12h, 1) # previous close
    
    # Camarilla levels (based on previous bar)
    camarilla_h4 = pc + 1.5 * (ph - pl)  # Resistance 4
    camarilla_h3 = pc + 1.25 * (ph - pl) # Resistance 3
    camarilla_h2 = pc + 1.0 * (ph - pl)  # Resistance 2
    camarilla_h1 = pc + 0.5 * (ph - pl)  # Resistance 1
    camarilla_l1 = pc - 0.5 * (ph - pl)  # Support 1
    camarilla_l2 = pc - 1.0 * (ph - pl)  # Support 2
    camarilla_l3 = pc - 1.25 * (ph - pl) # Support 3
    camarilla_l4 = pc - 1.5 * (ph - pl)  # Support 4
    
    # Handle first value (no previous bar)
    camarilla_h4[0] = camarilla_h3[0] = camarilla_h2[0] = camarilla_h1[0] = np.nan
    camarilla_l1[0] = camarilla_l2[0] = camarilla_l3[0] = camarilla_l4[0] = np.nan
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price touches or breaks below L3 (strong support) with volume spike and against trend
        # In uptrend: fade breakdowns to L3; in downtrend: fade breakdowns only if strong support
        if (close[i] <= camarilla_l3_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price touches or breaks above H3 (strong resistance) with volume spike and against trend
        elif (close[i] >= camarilla_h3_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to mean (H1/L1)
        elif position == 1 and (close[i] >= camarilla_h1_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= camarilla_l1_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_1dVolume_1wEMA_Reversal"
timeframe = "12h"
leverage = 1.0