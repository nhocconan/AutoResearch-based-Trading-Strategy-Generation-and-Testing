#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v2
Daily strategy using KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
Enters long when KAMA trend is up, RSI > 50, and choppy market (CHOP > 61.8).
Enters short when KAMA trend is down, RSI < 50, and choppy market (CHOP > 61.8).
Exits when opposite conditions occur.
Uses weekly ADX as trend filter to avoid whipsaws in weak trends.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        er = np.zeros_like(close)
        for i in range(len(close)):
            if i >= er_length:
                change_sum = np.sum(change[i-er_length+1:i+1])
                volatility_sum = np.sum(volatility[i-er_length+1:i+1])
                er[i] = change_sum / (volatility_sum + 1e-10)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    kama_up = kama_vals > np.roll(kama_vals, 1)
    kama_down = kama_vals < np.roll(kama_vals, 1)
    
    # === RSI (Relative Strength Index) ===
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # === Choppiness Index ===
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(window=length, min_periods=length).mean().values
        
        hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
        ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
        chop = 100 * np.log10((atr * length) / (hh - ll + 1e-10)) / np.log10(length)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    choppy = chop > 61.8  # Choppy/ranging market
    
    # === Weekly ADX for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components (14-period)
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(low_1w)
    plus_dm[1:] = np.maximum(high_1w[1:] - high_1w[:-1], 0)
    minus_dm[1:] = np.maximum(low_1w[:-1] - low_1w[1:], 0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1w[0] - low_1w[0]
    
    atr_w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_w * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_w * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: only trade when weekly ADX > 20 (avoid extremely weak trends)
        trending_enough = adx_1w_aligned[i] > 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA up, RSI > 50, choppy market, and sufficient trend strength
            if kama_up[i] and rsi_vals[i] > 50 and choppy[i] and trending_enough:
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA down, RSI < 50, choppy market, and sufficient trend strength
            elif kama_down[i] and rsi_vals[i] < 50 and choppy[i] and trending_enough:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA down OR RSI < 50
            if kama_down[i] or rsi_vals[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up OR RSI > 50
            if kama_up[i] or rsi_vals[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0