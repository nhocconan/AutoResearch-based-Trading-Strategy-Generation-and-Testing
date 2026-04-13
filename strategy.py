#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter Strategy.
Trades in direction of Kaufman Adaptive Moving Average (KAMA) when RSI is not extreme,
and only during choppy markets (Choppiness Index > 61.8) to avoid false signals in strong trends.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
Uses 1h for trend filter to avoid counter-trend trades during strong moves.
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
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
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
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Choppiness Index (14-period)
    chop_len = 14
    atr = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = np.zeros(n)
    for i in range(chop_len, n):
        atr_sum[i] = np.sum(tr[i-chop_len+1:i+1])
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(chop_len-1, n):
        max_high[i] = np.max(high[i-chop_len+1:i+1])
        min_low[i] = np.min(low[i-chop_len+1:i+1])
    
    chop = np.zeros(n)
    for i in range(chop_len-1, n):
        if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(chop_len)
        else:
            chop[i] = 50
    
    # Get 1h data for trend filter (avoid counter-trend trades)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    # Simple trend: price above/below 20-period EMA
    ema_20 = pd.Series(close_1h).ewm(span=20, adjust=False).mean().values
    uptrend_1h = close_1h > ema_20
    downtrend_1h = close_1h < ema_20
    
    uptrend_1h_aligned = align_htf_to_ltf(prices, df_1h, uptrend_1h.astype(float))
    downtrend_1h_aligned = align_htf_to_ltf(prices, df_1h, downtrend_1h.astype(float))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(uptrend_1h_aligned[i]) or np.isnan(downtrend_1h_aligned[i])):
            continue
        
        # KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI not extreme (avoid overbought/oversold)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Chop filter: only trade in choppy markets
        choppy = chop[i] > 61.8
        
        # Entry conditions
        long_entry = kama_up and rsi_not_overbought and choppy and uptrend_1h_aligned[i] > 0.5
        short_entry = kama_down and rsi_not_oversold and choppy and downtrend_1h_aligned[i] > 0.5
        
        # Exit when KAMA direction changes or RSI extreme
        exit_long = position == 1 and (not kama_up or rsi[i] >= 70)
        exit_short = position == -1 and (not kama_down or rsi[i] <= 30)
        
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
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_kama_rsi_chop_filter"
timeframe = "1d"
leverage = 1.0