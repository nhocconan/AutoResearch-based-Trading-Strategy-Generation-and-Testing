#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Strategy: 1d KAMA trend with RSI filter and Chop regime filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses KAMA (Kaufman Adaptive Moving Average) to capture adaptive trend direction, combined with RSI for momentum confirmation and Chop index to avoid ranging markets. Designed to work in both bull and bear markets by only taking trades when trend is strong (KAMA divergence) and market is not choppy (Chop < 61.8). Weekly trend filter ensures alignment with higher timeframe momentum. Target: 20-60 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Calculate ER using rolling window
    er = np.zeros(n)
    for i in range(er_len, n):
        price_change = np.abs(close[i] - close[i-er_len])
        price_volatility = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    rsi_len = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < rsi_len:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_len-1) + gain[i]) / rsi_len
            avg_loss[i] = (avg_loss[i-1] * (rsi_len-1) + loss[i]) / rsi_len
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop index calculation
    chop_len = 14
    atr = np.zeros(n)
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATR using smoothed moving average
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (chop_len-1) + tr[i]) / chop_len
    
    # Calculate Chop index
    sum_tr = np.zeros(n)
    max_h = np.zeros(n)
    min_l = np.zeros(n)
    for i in range(chop_len-1, n):
        sum_tr[i] = np.sum(tr[i-chop_len+1:i+1])
        max_h[i] = np.max(high[i-chop_len+1:i+1])
        min_l[i] = np.min(low[i-chop_len+1:i+1])
        range_max_min = max_h[i] - min_l[i]
        if range_max_min > 0:
            chop = 100 * np.log10(sum_tr[i] / range_max_min) / np.log10(chop_len)
        else:
            chop = 50
        if i == chop_len-1:
            chop_vals = chop
        else:
            chop_vals = np.append(chop_vals, chop) if 'chop_vals' in locals() else chop
    
    chop_final = np.full(n, 50.0)
    if 'chop_vals' in locals():
        chop_final[chop_len-1:len(chop_vals)+chop_len-1] = chop_vals
    
    # Weekly trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_final[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend condition: price relative to KAMA
        above_kama = price_close > kama[i]
        below_kama = price_close < kama[i]
        
        # Momentum condition: RSI
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        
        # Regime condition: Chop < 61.8 (trending market)
        not_choppy = chop_final[i] < 61.8
        
        # Weekly trend filter
        weekly_uptrend = price_close > ema_20_1w_aligned[i]
        weekly_downtrend = price_close < ema_20_1w_aligned[i]
        
        # Long: price above KAMA, RSI not overbought, not choppy, weekly uptrend
        long_signal = above_kama and not rsi_overbought and not_choppy and weekly_uptrend
        
        # Short: price below KAMA, RSI not oversold, not choppy, weekly downtrend
        short_signal = below_kama and not rsi_oversold and not_choppy and weekly_downtrend
        
        # Exit conditions
        exit_long = position == 1 and (price_close < kama[i] or chop_final[i] > 61.8)
        exit_short = position == -1 and (price_close > kama[i] or chop_final[i] > 61.8)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals