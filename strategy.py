#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_WeeklyTrend_ATR_Exit_v2
Hypothesis: Camarilla R1/S1 breakouts on 1d with weekly EMA50 trend filter and ATR-based trailing stop (2.5x ATR) captures strong institutional moves while minimizing whipsaws. Uses discrete sizing (0.25) to limit fee churn. Designed for 1d timeframe to achieve 7-25 trades/year with proper risk control in both bull and bear markets.
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 1d (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # ATR(14) for volatility and trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long positions
    lowest_since_entry = 0.0   # for short positions
    base_size = 0.25
    
    # Warmup: max of EMA(50), ATR(14)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_50_1w_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price vs weekly EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above R1 with weekly uptrend
        long_condition = (close_val > r1_val) and uptrend
        # Short: price CLOSES below S1 with weekly downtrend
        short_condition = (close_val < s1_val) and downtrend
        
        # Exit conditions
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            # Update highest price since entry
            if close_val > highest_since_entry:
                highest_since_entry = close_val
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if close_val <= highest_since_entry - 2.5 * atr_val:
                long_exit = True
        elif position == -1:  # Short position
            # Update lowest price since entry
            if close_val < lowest_since_entry:
                lowest_since_entry = close_val
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close_val >= lowest_since_entry + 2.5 * atr_val:
                short_exit = True
        
        # Entry logic
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_entry = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            lowest_since_entry = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_entry = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_ATR_Exit_v2"
timeframe = "1d"
leverage = 1.0