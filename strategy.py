#!/usr/bin/env python3
"""
Daily KAMA + RSI + Chop Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise,
providing smoother trend signals. Combined with RSI overbought/oversold levels
and a chop filter (Choppiness Index > 61.8 = ranging), this strategy avoids
whipsaws in sideways markets while capturing trends in both bull and bear regimes.
Targets 10-25 trades/year on daily timeframe.
"""
name = "daily_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter - call ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === KAMA calculation (10-period ER, 2/30 SC) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    abs_sum = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will fix properly
    # Correct ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/30 - 2/10) + 2/10) ** 2  # FC = 2/10, SC = 2/30
    
    # KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of TR over 14 periods
    tr_sum14 = np.convolve(tr, np.ones(14), 'same')  # approximate, will refine
    # Better calculation
    tr_sum14 = np.zeros(n)
    for i in range(14, n):
        tr_sum14[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros(n)
    ll = np.zeros(n)
    for i in range(14, n):
        hh[i] = np.max(high[i-13:i+1])
        ll[i] = np.min(low[i-13:i+1])
    
    # Chop = 100 * log10(sumTR14 / (HH - LL)) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(tr_sum14[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 100  # max choppy when range is zero
    
    # === Weekly trend filter (EMA 20 > EMA 50) ===
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = ema20_1w > ema50_1w
    
    # Align weekly data to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(weekly_uptrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI overbought OR chop too low (trending)
            if kama[i] < close[i] or rsi[i] > 70 or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI oversold OR chop too low
            if kama[i] > close[i] or rsi[i] < 30 or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in choppy markets (Chop > 61.8 = ranging)
            if chop[i] > 61.8:
                # Long: price above KAMA and RSI oversold
                if close[i] > kama[i] and rsi[i] < 30 and weekly_up:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA and RSI overbought
                elif close[i] < kama[i] and rsi[i] > 70 and not weekly_up:
                    position = -1
                    signals[i] = -0.25
    
    return signals