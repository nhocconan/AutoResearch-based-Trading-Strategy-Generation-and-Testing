#!/usr/bin/env python3
"""
1d KAMA Direction with RSI Momentum Filter and Chop Regime
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. Combined with RSI momentum and Chop filter to avoid whipsaws in ranging markets. Works in both bull and bear by following adaptive trend with momentum confirmation. Targets 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w KAMA(30) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Efficiency Ratio for KAMA
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    abs_change_1w = np.diff(close_1w, prepend=close_1w[0])
    er_num_1w = np.abs(close_1w - np.roll(close_1w, 30))
    er_den_1w = np.sum(np.lib.stride_tricks.sliding_window_view(np.abs(abs_change_1w), 30), axis=1)
    er_den_1w = np.concatenate([np.full(29, np.nan), er_den_1w])
    er_1w = np.divide(er_num_1w, er_den_1w, out=np.full_like(er_num_1w, np.nan), where=er_den_1w!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_1w = (er_1w * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[30] = close_1w[30]
    for i in range(31, len(close_1w)):
        if not np.isnan(sc_1w[i]):
            kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d KAMA(30) for entry signal
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.diff(close, prepend=close[0])
    er_num = np.abs(close - np.roll(close, 30))
    er_den = np.sum(np.lib.stride_tricks.sliding_window_view(np.abs(abs_change), 30), axis=1)
    er_den = np.concatenate([np.full(29, np.nan), er_den])
    er = np.divide(er_num, er_den, out=np.full_like(er_num, np.nan), where=er_den!=0)
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 /(30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[30] = close[30]
    for i in range(31, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop Index(14) for regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(np.lib.stride_tricks.sliding_window_view(tr, 14), axis=1) / 
                          (np.log10(max_high - min_low) * 14)) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop])
    
    # Volume filter (>1.2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR chop too high (range)
            if (close[i] <= kama[i] or chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR chop too high (range)
            if (close[i] >= kama[i] or chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above KAMA, uptrend on 1w, RSI > 50, low chop
            if (close[i] > kama[i] and 
                close[i] > kama_1w_aligned[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA, downtrend on 1w, RSI < 50, low chop
            elif (close[i] < kama[i] and 
                  close[i] < kama_1w_aligned[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals