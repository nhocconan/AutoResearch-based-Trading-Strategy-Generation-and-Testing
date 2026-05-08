#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Breakout_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # === Weekly Pivot Points ===
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = high_w[0]
    prev_low_w[0] = low_w[0]
    prev_close_w[0] = close_w[0]
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    range_w = prev_high_w - prev_low_w
    
    r1_w = pivot_w + (range_w * 1.1 / 12)
    s1_w = pivot_w - (range_w * 1.1 / 12)
    r2_w = pivot_w + (range_w * 1.1 / 6)
    s2_w = pivot_w - (range_w * 1.1 / 6)
    
    # Align weekly pivot levels to 6h timeframe
    r1_w_6h = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_6h = align_htf_to_ltf(prices, df_w, s2_w)
    
    # === 6h EMA50 for trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h Volume filter: current volume > 1.5x 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or np.isnan(r2_w_6h[i]) or np.isnan(s2_w_6h[i]) or
            np.isnan(ema50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout above R2 with trend and volume
            long_cond = (close[i] > r2_w_6h[i] and 
                        close[i] > ema50[i] and
                        volume[i] > vol_ma20[i] * 1.5)
            
            # Breakdown below S2 with trend and volume
            short_cond = (close[i] < s2_w_6h[i] and 
                         close[i] < ema50[i] and
                         volume[i] > vol_ma20[i] * 1.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below S1 or trend reversal
            if close[i] < s1_w_6h[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above R1 or trend reversal
            if close[i] > r1_w_6h[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot-based breakout strategy on 6h timeframe. Enters on
# breakouts above R2 or below S2 only when aligned with 6h EMA50 trend and
# confirmed by elevated volume (>1.5x 20-period average). Exits when price
# returns to S1/R1 or trend reverses. Uses weekly pivot points for institutional
# reference levels that work in both bull and bear markets. Targets 50-150
# trades over 4 years to minimize fee drag. Uses discrete sizing (0.25) to
# reduce churn. Weekly pivots avoid noise of lower timeframe levels.