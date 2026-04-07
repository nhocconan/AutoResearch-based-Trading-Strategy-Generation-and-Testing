#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA Trend with Weekly RSI Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both bull and bear markets.
# Weekly RSI filter ensures we only trade when momentum is aligned with higher timeframe.
# Long when KAMA upward and weekly RSI > 50; short when KAMA downward and weekly RSI < 50.
# Volatility filter (ATR) prevents whipsaws in ranging markets.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "daily_kama_trend_weekly_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for RSI filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    weekly_rsi = np.where(avg_loss == 0, 100, 0)
    weekly_rsi = np.where((avg_gain != 0) & (avg_loss != 0), 100 - (100 / (1 + rs)), weekly_rsi)
    
    # Align weekly RSI to daily
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    # Calculate KAMA(10, 2, 30) on daily
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # needs correction
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_abs_change = 0
            for j in range(1, 11):
                sum_abs_change += np.abs(close[i-j+1] - close[i-j])
            if sum_abs_change > 0:
                er[i] = price_change / sum_abs_change
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volatility filter: ATR > 50-day average ATR
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > (0.5 * atr_ma)  # Only trade when volatility is above half its MA
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(weekly_rsi_aligned[i]) or 
            np.isnan(atr_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down or volatility drops
            if kama[i] < kama[i-1] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: KAMA turns up or volatility drops
            if kama[i] > kama[i-1] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: KAMA upward and weekly RSI > 50
            if kama[i] > kama[i-1] and weekly_rsi_aligned[i] > 50 and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: KAMA downward and weekly RSI < 50
            elif kama[i] < kama[i-1] and weekly_rsi_aligned[i] < 50 and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals