#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Chop_Filter_v1
Hypothesis: On 1d timeframe, KAMA trend direction combined with RSI momentum and Choppiness Index regime filter captures trending moves while avoiding chop. KAMA adapts to market noise, RSI filters for momentum strength, and Choppiness Index (>61.8 = chop, <38.2 = trend) ensures we only trade in trending regimes. This reduces false signals in sideways markets, improving performance in both bull and bear cycles. Target: 20-50 trades over 4 years.
"""
name = "1d_KAMA_Trend_RSI_Chop_Filter_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 9 values where diff isn't available
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder smoothing
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr = np.full_like(close, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr_sum = np.full_like(close, np.nan)
    for i in range(13, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    chop = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # 1-week EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align KAMA, RSI, Chop to daily timeframe (already aligned as we calculated on close)
    # But we need to ensure no look-ahead: KAMA, RSI, Chop use only past data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # after KAMA/RSI/Chop warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI > 50 (momentum) + Chop < 38.2 (trending) + weekly uptrend
            if close[i] > kama[i] and rsi[i] > 50 and chop[i] < 38.2 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI < 50 (momentum) + Chop < 38.2 (trending) + weekly downtrend
            elif close[i] < kama[i] and rsi[i] < 50 and chop[i] < 38.2 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: trend change or chop regime
            if position == 1:
                if close[i] < kama[i] or chop[i] > 61.8:  # trend broken or too choppy
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or chop[i] > 61.8:  # trend broken or too choppy
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals