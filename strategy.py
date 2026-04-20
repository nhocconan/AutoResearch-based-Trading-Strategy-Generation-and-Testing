#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Adaptive_Breakout_With_Pullback"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate pivot points (R1/S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1w: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 4h: ATR for volatility and pullback detection ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h: 20-period EMA for pullback entry ===
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        ema50_1w = ema_50_1w_aligned[i]
        current_atr = atr[i]
        current_close = close[i]
        current_ema20 = ema_20[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1) or np.isnan(s1) or np.isnan(ema50_1w) or 
            np.isnan(current_atr) or np.isnan(current_ema20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 4h average volume ===
        if i >= 20:
            vol_ma = np.mean(prices['volume'].iloc[i-20:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        # === Trend filter: price above/below weekly EMA50 ===
        uptrend = current_close > ema50_1w
        downtrend = current_close < ema50_1w
        
        if position == 0:
            # Long conditions:
            # 1. Pullback to EMA20 in uptrend
            # 2. Bounce above EMA20 with volume
            # 3. Price above daily S1 (support)
            if (uptrend and 
                current_close > current_ema20 and 
                close[i-1] <= current_ema20 and  # crossed above EMA20
                vol_condition and 
                current_close > s1):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Pullback to EMA20 in downtrend
            # 2. Rejection below EMA20 with volume
            # 3. Price below daily R1 (resistance)
            elif (downtrend and 
                  current_close < current_ema20 and 
                  close[i-1] >= current_ema20 and  # crossed below EMA20
                  vol_condition and 
                  current_close < r1):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Break below daily S1 (support broken)
            # 2. Pullback to EMA20 fails (price drops back below)
            # 3. ATR-based stop loss
            if (current_close < s1 or 
                current_close < current_ema20 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Break above daily R1 (resistance broken)
            # 2. Pullback to EMA20 fails (price rises back above)
            # 3. ATR-based stop loss
            if (current_close > r1 or 
                current_close > current_ema20 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals