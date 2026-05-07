#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Chop_Filter_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    trend_up = close > ema34_1d_aligned
    trend_down = close < ema34_1d_aligned
    
    # KAMA calculation
    er = np.zeros(n)
    change = np.abs(close[10:] - close[:-10])
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    volatility = np.concatenate([np.zeros(10), volatility])
    er[10:] = change / volatility
    er[volatility == 0] = 0
    er = np.clip(er, 0, 1)
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_dir = np.zeros(n)
    kama_dir[1:] = np.sign(kama[1:] - kama[:-1])
    
    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50
    
    # Choppiness Index (14-period)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr = np.zeros(n)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    highest = np.zeros(n)
    lowest = np.zeros(n)
    highest[0] = high[0]
    lowest[0] = low[0]
    for i in range(1, n):
        highest[i] = max(highest[i-1], high[i])
        lowest[i] = min(lowest[i-1], low[i])
    chop = np.zeros(n)
    for i in range(13, n):
        sum_atr = np.sum(atr[i-13:i+1])
        if highest[i] == lowest[i]:
            chop[i] = 100
        else:
            chop[i] = 100 * np.log10(sum_atr / (highest[i] - lowest[i])) / np.log10(14)
    chop[:13] = 50
    
    signals = np.zeros(n)
    position = 0
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~12 hours
    
    start_idx = max(30, 14, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending)
            if kama_dir[i] > 0 and rsi[i] > 50 and chop[i] < 61.8 and trend_up[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA down, RSI < 50, chop < 61.8 (trending)
            elif kama_dir[i] < 0 and rsi[i] < 50 and chop[i] < 61.8 and trend_down[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: KAMA down OR chop > 61.8 (range) OR trend down
            if kama_dir[i] < 0 or chop[i] > 61.8 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up OR chop > 61.8 (range) OR trend up
            if kama_dir[i] > 0 or chop[i] > 61.8 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA direction + RSI + Chop filter on 4h with 1d trend filter.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI > 50 for long, < 50 for short ensures momentum alignment.
# Chop < 61.8 filters for trending markets only, avoiding whipsaws in ranges.
# 1d EMA34 trend filter ensures alignment with higher timeframe trend.
# Works in bull markets (captures uptrends) and bear markets (captures downtrends).
# Target: 20-50 trades/year to minimize fee drag. Position size 0.25 manages risk.