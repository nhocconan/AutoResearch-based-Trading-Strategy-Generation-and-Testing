#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R1 level with 1d EMA50 uptrend and volume > 2.0x 20-bar average
# Short when price breaks below 4h Camarilla S1 level with 1d EMA50 downtrend and volume > 2.0x 20-bar average
# Exit via close reversal: long exit when price < 4h Camarilla Pivot level, short exit when price > 4h Camarilla Pivot level
# Uses 4h Camarilla pivots for structure, 1d EMA50 for trend filter, volume spike for confirmation
# Discrete sizing 0.20 to minimize fee drag. Target: 60-150 total trades over 4 years = 15-37/year.

name = "1h_Camarilla_R1S1_1dEMA50_Volume_CloseExit_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla pivot levels (using prior 4h bar's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h bar's OHLC for Camarilla calculation
    ph_4h = df_4h['high'].shift(1).values  # prior 4h high
    pl_4h = df_4h['low'].shift(1).values   # prior 4h low
    pc_4h = df_4h['close'].shift(1).values # prior 4h close
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, Pivot = C
    camarilla_r1_4h = pc_4h + (ph_4h - pl_4h) * 1.1 / 12
    camarilla_s1_4h = pc_4h - (ph_4h - pl_4h) * 1.1 / 12
    camarilla_pivot_4h = pc_4h
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA50 and volume calculations)
    start_idx = 50  # EMA50 needs 50 bars, plus buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Camarilla R1 with 1d EMA50 uptrend and volume spike
            if close[i] > camarilla_r1_aligned[i] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Camarilla S1 with 1d EMA50 downtrend and volume spike
            elif close[i] < camarilla_s1_aligned[i] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price closes below 4h Camarilla Pivot level
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price closes above 4h Camarilla Pivot level
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals