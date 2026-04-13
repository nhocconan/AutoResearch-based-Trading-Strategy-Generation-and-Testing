#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from weekly high/low/close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: P = (H + L + C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance 1: R1 = 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # Support 1: S1 = 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    
    # Calculate 14-period RSI on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate 20-period ATR on daily for volatility filter
    tr1 = np.maximum(high_1d, np.roll(close_1d, 1))
    tr1[0] = high_1d[0]
    tr2 = np.maximum(tr1, low_1d)
    tr3 = np.minimum(tr2, np.roll(close_1d, 1))
    tr3[0] = low_1d[0]
    tr = np.maximum(tr2, tr3) - np.minimum(tr1, tr2)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(250, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to weekly pivot
        above_pivot = close[i] > pivot_1w_aligned[i]
        below_pivot = close[i] < pivot_1w_aligned[i]
        
        # RSI conditions: avoid extremes
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        
        # Entry conditions: bounce off weekly support/resistance with RSI confirmation
        long_entry = (close[i] <= s1_1w_aligned[i] * 1.02) and rsi_not_oversold and vol_filter
        short_entry = (close[i] >= r1_1w_aligned[i] * 0.98) and rsi_not_overbought and vol_filter
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = position == 1 and (rsi_14_aligned[i] > 70 or close[i] >= pivot_1w_aligned[i] * 0.99)
        exit_short = position == -1 and (rsi_14_aligned[i] < 30 or close[i] <= pivot_1w_aligned[i] * 1.01)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6d_wp_rsi_vol_filter"
timeframe = "6h"
leverage = 1.0