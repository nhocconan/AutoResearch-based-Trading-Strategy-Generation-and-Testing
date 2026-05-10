#!/usr/bin/env python3
# 6h_RSI_Divergence_With_Trend_and_Volume
# Hypothesis: RSI divergence (bullish/bearish) on 6h combined with 1d EMA50 trend and volume spike
# captures reversal points in both bull and bear markets. Divergence signals exhaustion,
# trend filter ensures trades follow higher timeframe momentum, volume confirms conviction.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "6h_RSI_Divergence_With_Trend_and_Volume"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI (14), EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # RSI divergence detection (look back 3 bars for swing high/low)
        bullish_div = False
        bearish_div = False
        
        if i >= 3:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if low[i] < low[i-3] and rsi[i] > rsi[i-3]:
                bullish_div = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            if high[i] > high[i-3] and rsi[i] < rsi[i-3]:
                bearish_div = True
        
        if position == 0:
            # Long entry: bullish divergence + uptrend + volume
            if bullish_div and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish divergence + downtrend + volume
            elif bearish_div and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish divergence or trend breaks
            if bearish_div or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish divergence or trend breaks
            if bullish_div or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals