#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-14:i+1])
    
    # Align weekly indicators to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 6h ATR(14) for volatility breakout
    tr1_6h = np.abs(high - low)
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr1_6h[0] = tr2_6h[0] = tr3_6h[0] = np.nan
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr_6h[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility breakout: current 6h ATR > 1.5 * weekly ATR
        vol_breakout = atr_6h[i] > 1.5 * atr_1w_aligned[i]
        
        # Trend filter: price above/below weekly EMA(34)
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Entry conditions
        long_entry = vol_breakout and above_ema
        short_entry = vol_breakout and below_ema
        
        # Exit conditions: volatility contraction or trend reversal
        long_exit = (atr_6h[i] < atr_1w_aligned[i]) or (close[i] < ema_34_1w_aligned[i])
        short_exit = (atr_6h[i] < atr_1w_aligned[i]) or (close[i] > ema_34_1w_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_vol_breakout_ema_filter_v1"
timeframe = "6h"
leverage = 1.0