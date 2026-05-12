#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyPullback_Entry"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 1-Week Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== Daily Pullback Entry =====
    # Daily EMA(20) for pullback reference
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily RSI(14) for oversold/overbought
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== Volume Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(vol_surge[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + Daily pullback to EMA20 + RSI oversold + Volume surge
            if (close[i] > ema50_1w_aligned[i] and  # Weekly uptrend
                close[i] <= ema20[i] * 1.01 and    # Near or slightly below daily EMA20
                rsi[i] < 30 and                    # Oversold
                vol_surge[i]):                     # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + Daily bounce to EMA20 + RSI overbought + Volume surge
            elif (close[i] < ema50_1w_aligned[i] and   # Weekly downtrend
                  close[i] >= ema20[i] * 0.99 and      # Near or slightly above daily EMA20
                  rsi[i] > 70 and                      # Overbought
                  vol_surge[i]):                       # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weekly trend reversal OR RSI overbought
            if close[i] < ema50_1w_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weekly trend reversal OR RSI oversold
            if close[i] > ema50_1w_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals