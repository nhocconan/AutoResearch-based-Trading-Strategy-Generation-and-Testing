# #!/usr/bin/env python3
# [24978] 1d_1w_kama_rsi_chop_v1
# Hypothesis: Daily KAMA (trend-following) combined with RSI momentum and weekly Chop index regime filter.
# Long when KAMA is rising, RSI > 50, and weekly Chop < 61.8 (trending market).
# Short when KAMA is falling, RSI < 50, and weekly Chop < 61.8 (trending market).
# Uses Chop index to avoid ranging markets where trend strategies fail.
# Exit when KAMA changes direction or Chop > 61.8 (range detected).
# Designed for low trade frequency (<25/year) to minimize fee drag in bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Chop index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Chop index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    chop = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w = np.zeros(len(df_1w))
        for i in range(1, len(df_1w)):
            tr = max(
                high_1w[i] - low_1w[i],
                abs(high_1w[i] - df_1w['close'].values[i-1]),
                abs(low_1w[i] - df_1w['close'].values[i-1])
            )
            atr_1w[i] = 0.93 * atr_1w[i-1] + 0.07 * tr  # Wilder smoothing
        
        # Calculate highest high and lowest low over 14 periods
        hh_1w = np.full(len(df_1w), np.nan)
        ll_1w = np.full(len(df_1w), np.nan)
        for i in range(13, len(df_1w)):
            hh_1w[i] = np.max(high_1w[i-13:i+1])
            ll_1w[i] = np.min(low_1w[i-13:i+1])
        
        # Chop = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
        sum_atr = np.full(len(df_1w), np.nan)
        for i in range(13, len(df_1w)):
            sum_atr[i] = np.sum(atr_1w[i-13:i+1])
            denominator = hh_1w[i] - ll_1w[i]
            if denominator > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / denominator) / np.log10(14)
    
    # Calculate daily KAMA (adaptive moving average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 1), axis=1)  # placeholder
    # Correct volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    er = np.zeros(n)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[14:i+1])
            avg_loss[i] = np.mean(loss[14:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100  # Avoid division by zero
    
    # Align weekly Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i-1]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        trending_market = chop_val < 61.8  # Chop < 61.8 = trending
        
        if position == 1:  # Long
            # Exit: KAMA turns down OR market becomes ranging
            if not kama_rising or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: KAMA turns up OR market becomes ranging
            if not kama_falling or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA rising, RSI > 50, trending market
            if kama_rising and rsi_above_50 and trending_market:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA falling, RSI < 50, trending market
            elif kama_falling and rsi_below_50 and trending_market:
                position = -1
                signals[i] = -0.25
    
    return signals