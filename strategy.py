#!/usr/bin/env python3
# 1h_RSI_Trend_Filter_Breakout
# Hypothesis: On 1h timeframe, use 4h trend direction via EMA50 and 1d RSI extremes for mean reversion entries.
# Long when 4h EMA50 up, 1d RSI < 30, and price breaks above 1h high of last 4 bars with volume > 1.5x average.
# Short when 4h EMA50 down, 1d RSI > 70, and price breaks below 1h low of last 4 bars with volume > 1.5x average.
# Exit when 4h EMA50 direction changes or RSI reverts to neutral (40-60).
# Designed for 15-30 trades/year to avoid overtrading and work in both bull and bear markets.

name = "1h_RSI_Trend_Filter_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d RSI for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Rolling high/low for breakout (4 periods)
    high_4 = np.full(n, np.nan)
    low_4 = np.full(n, np.nan)
    for i in range(4, n):
        high_4[i] = np.max(high[i-4:i])
        low_4[i] = np.min(low[i-4:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_4[i]) or np.isnan(low_4[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction: 1 if EMA rising, -1 if falling
        if i > start_idx:
            ema_trend = 1 if ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] else -1
        else:
            ema_trend = 0
        
        if position == 0:
            # Long: Uptrend, oversold RSI, breakout above recent high with volume
            if ema_trend == 1 and rsi_1d_aligned[i] < 30 and close[i] > high_4[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend, overbought RSI, breakout below recent low with volume
            elif ema_trend == -1 and rsi_1d_aligned[i] > 70 and close[i] < low_4[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Trend turns down or RSI returns to neutral
            if ema_trend == -1 or rsi_1d_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Trend turns up or RSI returns to neutral
            if ema_trend == 1 or rsi_1d_aligned[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals