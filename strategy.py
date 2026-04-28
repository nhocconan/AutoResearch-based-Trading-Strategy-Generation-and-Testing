#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop
Hypothesis: On daily timeframe, enter long when KAMA indicates uptrend, RSI < 40 (oversold), and choppy market (CHOP > 61.8); enter short when KAMA indicates downtrend, RSI > 60 (overbought), and choppy market. Exit when trend reverses. Uses weekly trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) to minimize fee decay in ranging/bear markets.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.maximum(volatility[10:], 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with 50
    rsi = np.concatenate([np.full(14, 50), rsi])
    
    # Daily Choppiness Index (CHOP)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if atr14[i] > 0:
            chop[i] = 100 * np.log10(highest_high[i] - lowest_low[i]) / np.log10(14) / atr14[i]
        else:
            chop[i] = 50
    
    # Align daily indicators (already aligned as we use prices directly)
    # KAMA, RSI, CHOP are already daily
    
    # Trend filter: weekly EMA50
    weekly_uptrend = close > ema_50_1w_aligned
    weekly_downtrend = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: KAMA up (price > KAMA), RSI oversold (<40), choppy market (CHOP > 61.8)
        long_entry = (close[i] > kama[i]) and (rsi[i] < 40) and (chop[i] > 61.8)
        # Short: KAMA down (price < KAMA), RSI overbought (>60), choppy market (CHOP > 61.8)
        short_entry = (close[i] < kama[i]) and (rsi[i] > 60) and (chop[i] > 61.8)
        
        # Exit when trend changes or opposite signal
        long_exit = (close[i] < kama[i]) or (rsi[i] > 60)
        short_exit = (close[i] > kama[i]) or (rsi[i] < 40)
        
        # Apply weekly trend filter - only take trades in direction of weekly trend
        if long_entry and weekly_uptrend[i]:
            signals[i] = 0.25
        elif short_entry and weekly_downtrend[i]:
            signals[i] = -0.25
        elif long_exit:
            signals[i] = 0.0
        elif short_exit:
            signals[i] = 0.0
        else:
            # Hold flat
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_RSI_Chop"
timeframe = "1d"
leverage = 1.0