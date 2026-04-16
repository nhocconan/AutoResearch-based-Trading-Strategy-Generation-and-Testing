#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA50 filter + volume confirmation
# Long when Williams %R(14) crosses above -50 (bullish momentum) AND price > 1d EMA50 AND volume > 1.5x 20-period avg volume
# Short when Williams %R(14) crosses below -50 (bearish momentum) AND price < 1d EMA50 AND volume > 1.5x 20-period avg volume
# Williams %R identifies momentum extremes and reversals; EMA50 ensures trend alignment; volume adds conviction
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA50 filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 60-period Volume Confirmation (10-period average for 6h: 10*6h = 60h ≈ 2.5 days) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_1d_aligned[i]
        williams_r_val = williams_r[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1] if i > 0 else -50
        crossed_above_50 = williams_r_prev <= -50 and williams_r_val > -50
        crossed_below_50 = williams_r_prev >= -50 and williams_r_val < -50
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Williams %R crosses above -50 AND price > EMA50 AND volume confirmation
            if crossed_above_50 and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Williams %R crosses below -50 AND price < EMA50 AND volume confirmation
            elif crossed_below_50 and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC ===
        elif position == 1:
            # Exit long when Williams %R crosses below -80 (overbought) OR price < EMA50
            if williams_r_val < -80 or price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses above -20 (oversold) OR price > EMA50
            if williams_r_val > -20 or price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dEMA50_Volume1.5x"
timeframe = "6h"
leverage = 1.0