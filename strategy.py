#!/usr/bin/env python3
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
    
    # === 1d data (HTF for ATR and levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate ATR(14) on 1d for volatility filter ===
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 12h data (HTF for EMA trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 12h EMA34 for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h price action: Calculate 12-period high/low for breakout ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_12h_max = pd.Series(high_12h).rolling(window=12, min_periods=12).max().values
    low_12h_min = pd.Series(low_12h).rolling(window=12, min_periods=12).min().values
    high_12h_max_aligned = align_htf_to_ltf(prices, df_12h, high_12h_max)
    low_12h_min_aligned = align_htf_to_ltf(prices, df_12h, low_12h_min)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(high_12h_max_aligned[i]) or 
            np.isnan(low_12h_min_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_val = atr_1d_aligned[i]
        ema_34_12h_val = ema_34_12h_aligned[i]
        high_break = high_12h_max_aligned[i]
        low_break = low_12h_min_aligned[i]
        
        # === VOLATILITY FILTER: Only trade when volatility is elevated ===
        # Use ATR relative to its 50-period average
        if i >= 50:  # Need enough history for ATR average
            atr_ma_50 = np.mean(np.trim_zeros(atr_1d_aligned[max(0, i-49):i+1]) or [0])
            if atr_ma_50 > 0:
                atr_ratio = atr_val / atr_ma_50
                # Only trade when volatility is above average (avoid low volatility chop)
                if atr_ratio < 0.8:
                    # Hold current position or stay flat
                    if position == 1:
                        signals[i] = 0.20
                    elif position == -1:
                        signals[i] = -0.20
                    else:
                        signals[i] = 0.0
                    continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 12-period low or ATR-based stop
            if price < low_break or price < (high_break - 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 12-period high or ATR-based stop
            if price > high_break or price > (low_break + 1.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 12-period high with trend filter
            if (price > high_break) and (price > ema_34_12h_val):
                signals[i] = 0.20
                position = 1
                continue
            
            # SHORT: Price breaks below 12-period low with trend filter
            elif (price < low_break) and (price < ema_34_12h_val):
                signals[i] = -0.20
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Volatility_Filtered_Breakout_EMA34"
timeframe = "12h"
leverage = 1.0