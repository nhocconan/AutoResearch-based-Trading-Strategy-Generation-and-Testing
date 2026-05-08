#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI divergence with 1w EMA trend filter.
# Uses RSI(14) for overbought/oversold conditions and 1w EMA(34) for trend.
# Long when RSI < 30 (oversold) and 1w EMA up; short when RSI > 70 (overbought) and 1w EMA down.
# In strong trends (1w EMA slope > 0 for long, < 0 for short), avoids counter-trend trades.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets via trend-following and bear markets via mean reversion in corrections.

name = "1d_RSI_Divergence_1wEMA"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_slope = np.diff(ema_34_1w, prepend=np.nan)
    trend_1w_up = ema_34_1w_slope > 0
    trend_1w_down = ema_34_1w_slope < 0
    
    # Align 1w EMA slope to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for RSI and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) and 1w EMA trending up
            if rsi[i] < 30 and trend_1w_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) and 1w EMA trending down
            elif rsi[i] > 70 and trend_1w_down_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI overbought (>70) or 1w EMA turns down
            if rsi[i] > 70 or not trend_1w_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI oversold (<30) or 1w EMA turns up
            if rsi[i] < 30 or not trend_1w_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals