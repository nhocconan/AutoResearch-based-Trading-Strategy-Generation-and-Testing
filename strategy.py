#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Choppiness Index regime filter with 1-day RSI extremes.
# Long when weekly Choppiness Index > 61.8 (range) AND daily RSI < 30 (oversold).
# Short when weekly Choppiness Index > 61.8 (range) AND daily RSI > 70 (overbought).
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit).
# Choppiness Index identifies ranging markets where mean reversion works; RSI captures extremes.
# Works in bull/bear by focusing on range-bound conditions which occur in both regimes.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WeeklyChop_RSI_Range"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Weekly Choppiness Index (14-period)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 14:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_w = np.full_like(close_w, np.nan)
    for i in range(14, len(tr)):
        atr_w[i] = np.nanmean(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_w = np.full_like(close_w, np.nan)
    ll_w = np.full_like(close_w, np.nan)
    for i in range(13, len(high_w)):
        hh_w[i] = np.max(high_w[i-13:i+1])
        ll_w[i] = np.min(low_w[i-13:i+1])
    
    # Chop = 100 * log10( sum(ATR14) / (HH - LL) ) / log10(14)
    chop_sum = np.full_like(close_w, np.nan)
    for i in range(14, len(atr_w)):
        chop_sum[i] = np.nansum(atr_w[i-13:i+1])
    
    denominator = hh_w - ll_w
    chop = np.full_like(close_w, np.nan)
    mask = (denominator > 0) & (~np.isnan(chop_sum)) & (~np.isnan(denominator))
    chop[mask] = 100 * np.log10(chop_sum[mask] / denominator[mask]) / np.log10(14)
    
    # Align weekly chop to 12h
    chop_aligned = align_htf_to_ltf(prices, df_w, chop)
    
    # Daily RSI (14-period)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 14:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    delta = np.diff(close_d, prepend=close_d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_d, np.nan)
    avg_loss = np.full_like(close_d, np.nan)
    for i in range(14, len(close_d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_d, rsi_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 1)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: chop > 61.8 (range) AND RSI < 30 (oversold)
            long_cond = (chop_aligned[i] > 61.8) and (rsi_aligned[i] < 30)
            # Short: chop > 61.8 (range) AND RSI > 70 (overbought)
            short_cond = (chop_aligned[i] > 61.8) and (rsi_aligned[i] > 70)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back to 40
            if rsi_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses back to 60
            if rsi_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals