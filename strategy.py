#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA direction + RSI + Chop filter + Weekly trend confirmation
# Long when KAMA rising, RSI < 50 (pullback in uptrend), Chop > 61.8 (range), weekly uptrend
# Short when KAMA falling, RSI > 50 (pullback in downtrend), Chop > 61.8 (range), weekly downtrend
# KAMA adapts to market noise, RSI captures mean reversion in range, Chop filters trending markets
# Weekly trend ensures alignment with higher timeframe momentum
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_KAMA_RSI_Chop_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA(14, 2, 30) - Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    # Corrected volatility calculation
    volatility = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-14):i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness Index(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.diff(close, prepend=close[0]))
    tr3 = np.abs(np.roll(close, 1) - close)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = np.zeros_like(close)
    for i in range(len(close)):
        start = max(0, i-14)
        atr_sum[i] = np.sum(atr[start:i+1]) if i >= start else 0
    chop = np.where(atr_sum != 0, 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14), 50)
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: KAMA rising, RSI < 50, Chop > 61.8 (range), weekly uptrend
            if kama_val > kama_prev and rsi_val < 50 and chop_val > 61.8 and ema34_1w_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI > 50, Chop > 61.8 (range), weekly downtrend
            elif kama_val < kama_prev and rsi_val > 50 and chop_val > 61.8 and ema34_1w_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or Chop < 38.2 (trending) or weekly trend down
            if kama_val < kama_prev or chop_val < 38.2 or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or Chop < 38.2 (trending) or weekly trend up
            if kama_val > kama_prev or chop_val < 38.2 or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals