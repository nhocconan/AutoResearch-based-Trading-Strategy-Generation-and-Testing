#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness regime filter
# Long when: KAMA rising (bullish trend), RSI < 30 (oversold), and choppy market (CHOP > 61.8)
# Short when: KAMA falling (bearish trend), RSI > 70 (overbought), and choppy market (CHOP > 61.8)
# Uses KAMA for adaptive trend, RSI for mean reversion in chop, Choppiness index to filter ranging markets.
# Works in bull/bear via trend filter (KAMA) + mean reversion in ranging conditions (RSI extremes).
# Timeframe: 1d (primary), HTF: 1w for higher timeframe trend context (not used in this version but available).

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |Change| / Sum(|daily changes|)
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    abs_change = np.abs(np.diff(close, n=1))
    abs_change = np.insert(abs_change, 0, 0)
    
    # 10-period ER
    er_num = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    er_num = np.insert(er_num, 0, [0]*10)  # pad first 10
    er_den = np.zeros(n)
    for i in range(n):
        if i >= 10:
            er_den[i] = np.sum(abs_change[i-9:i+1])  # sum of last 10 abs changes
        else:
            er_den[i] = np.sum(abs_change[:i+1]) if i > 0 else 1  # avoid div by zero
    er = np.where(er_den != 0, er_num / er_den, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, n=1)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP)
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            sum_atr[i] = np.sum(atr[:i+1])
        else:
            sum_atr[i] = np.sum(atr[i-13:i+1])
    
    # Max(high) - Min(low) over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        if i < 14:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = np.where(range_hl != 0, 100 * np.log10(sum_atr / range_hl) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        curr_kama = kama[i]
        curr_kama_prev = kama[i-1]
        curr_rsi = rsi[i]
        curr_chop = chop[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit when: KAMA turns down OR RSI > 50 (mean reversion) OR chop < 38.2 (trending)
            if (curr_kama < curr_kama_prev or
                curr_rsi > 50 or
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when: KAMA turns up OR RSI < 50 (mean reversion) OR chop < 38.2 (trending)
            if (curr_kama > curr_kama_prev or
                curr_rsi < 50 or
                curr_chop < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: KAMA rising, RSI oversold (<30), choppy market (CHOP > 61.8)
            if (curr_kama > curr_kama_prev and
                curr_rsi < 30 and
                curr_chop > 61.8):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling, RSI overbought (>70), choppy market (CHOP > 61.8)
            elif (curr_kama < curr_kama_prev and
                  curr_rsi > 70 and
                  curr_chop > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals