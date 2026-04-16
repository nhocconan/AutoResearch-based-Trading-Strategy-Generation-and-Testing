# USL Strategy
# Hypothesis: On 12h timeframe, price breaking above/below daily R1/S1 pivots with volume confirmation and price above/below daily EMA50 (trend filter) will capture sustained moves in both bull and bear markets.
# Uses 1d pivots and EMA50 for trend filter, volume > 1.5x 20-period MA for confirmation.
# Target: 15-30 trades/year on 12h (60-120 total over 4 years) to avoid fee drag.
# Position size: 0.25 (25% of capital) to limit drawdown during extended moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (HTF for pivot levels and trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Calculate 1d Pivot Points (Standard) ===
    # Using previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Standard pivot formula
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1 = 2 * pivot - prev_low_1d
    s1 = 2 * pivot - prev_high_1d
    r2 = pivot + (prev_high_1d - prev_low_1d)
    s2 = pivot - (prev_high_1d - prev_low_1d)
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation (12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S1 (stop) or hits R2 (target)
            if price < s1_val or price > r2_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R1 (stop) or hits S2 (target)
            if price > r1_val or price < s2_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume AND above daily EMA50 (uptrend)
            if (price > r1_val) and (price > ema_50_1d_val) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume AND below daily EMA50 (downtrend)
            elif (price < s1_val) and (price < ema_50_1d_val) and (vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "USL_12h_Pivot_R1_S1_Breakout_Volume_EMA50"
timeframe = "12h"
leverage = 1.0