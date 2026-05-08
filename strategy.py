# 12h_1w_Pivot_R1_S1_Trend_Filter
# Hypothesis: 12h price crossing above/below weekly R1/S1 with volume confirmation
# and 1-week trend filter. Uses institutional weekly pivot levels for structure.
# Works in bull markets via breakout continuation and bear markets via mean reversion
# at pivot levels when price rejects weekly resistance/support.
# Targets 20-50 trades/year to minimize fee drag. Uses discrete sizing (0.25).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Pivot_R1_S1_Trend_Filter"
timeframe = "12h"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly high/low/close for pivot calculation ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values (shift by 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # First value: use current week's values to avoid look-ahead
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Weekly pivot point (standard calculation)
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # R1 and S1 levels (most significant)
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)  # Standard R1
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)  # Standard S1
    
    # Align to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1-week EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume confirmation: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend: above/below weekly EMA50
            uptrend = close[i] > ema50_12h[i]
            downtrend = close[i] < ema50_12h[i]
            
            if uptrend:
                # Uptrend: look for breakout above R1 with volume
                long_cond = (close[i] > r1_12h[i] and 
                            volume[i] > vol_ma20[i])
                if long_cond:
                    signals[i] = 0.25
                    position = 1
            elif downtrend:
                # Downtrend: look for breakdown below S1 with volume
                short_cond = (close[i] < s1_12h[i] and 
                             volume[i] > vol_ma20[i])
                if short_cond:
                    signals[i] = -0.25
                    position = -1
            # No action in transition or without volume
        elif position == 1:
            # Long exit: price breaks below S1 or trend reverses
            exit_cond = (close[i] < s1_12h[i] or 
                        close[i] < ema50_12h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend reverses
            exit_cond = (close[i] > r1_12h[i] or 
                        close[i] > ema50_12h[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals