#!/usr/bin/env python3
# 1d_kama_rsi_v1
# Hypothesis: 1d KAMA direction + RSI + chop filter captures trend continuation in major moves while avoiding chop. Works in bull/bear by following KAMA trend. Uses tight entry conditions to limit trades (~10-25/year) and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for chop filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.maximum(np.abs(low_1w[1:] - close_1w[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop filter: ADX < 25 = choppy (avoid trend following)
    # We want trending markets: ADX >= 25
    chop_filter = adx >= 25
    
    # Align 1w chop filter to 1d
    chop_filter_aligned = align_htf_to_ltf(prices, df_1w, chop_filter)
    
    # KAMA on 1d
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    er = np.concatenate([[np.nan]*10, er])
    
    # Smoothing constants
    fastest = 2/(2+1)
    slowest = 2/(30+1)
    sc = (er * (fastest - slowest) + slowest)**2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) on 1d
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < KAMA OR RSI < 40
            if (close[i] < kama[i]) or (rsi[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > KAMA OR RSI > 60
            if (close[i] > kama[i]) or (rsi[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > KAMA + RSI > 50 + trending market (ADX >= 25)
            if (close[i] > kama[i]) and (rsi[i] > 50) and chop_filter_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price < KAMA + RSI < 50 + trending market (ADX >= 25)
            elif (close[i] < kama[i]) and (rsi[i] < 50) and chop_filter_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals