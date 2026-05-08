#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h momentum strategy using 1w EMA(8) trend filter + 1d RSI(14) for entry timing.
# Long when 1w trend up and 1d RSI < 30 (oversold in uptrend).
# Short when 1w trend down and 1d RSI > 70 (overbought in downtrend).
# Uses volume confirmation (1.5x 20-period volume average) to avoid false signals.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and cost.

name = "6h_1wEMA8_1dRSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(8) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_8_1w = close_1w_series.ewm(span=8, adjust=False, min_periods=8).mean().values
    trend_up = ema_8_1w[1:] > ema_8_1w[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1w index
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # First value neutral
    
    # Volume confirmation: 1.5x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align 1w and 1d indicators to 6h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 1w uptrend + 1d RSI oversold + volume confirmation
            if (trend_up_aligned[i] > 0.5 and  # 1w uptrend
                rsi_aligned[i] < 30 and        # 1d RSI oversold
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: 1w downtrend + 1d RSI overbought + volume confirmation
            elif (trend_up_aligned[i] <= 0.5 and  # 1w downtrend
                  rsi_aligned[i] > 70 and       # 1d RSI overbought
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w trend turns down OR RSI overbought
            if (trend_up_aligned[i] <= 0.5 or  # 1w downtrend
                rsi_aligned[i] > 70):          # 1d RSI overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w trend turns up OR RSI oversold
            if (trend_up_aligned[i] > 0.5 or   # 1w uptrend
                rsi_aligned[i] < 30):          # 1d RSI oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals