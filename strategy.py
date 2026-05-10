#!/usr/bin/env python3
# 4h_RSI_Divergence_With_1dTrend_VolumeFilter
# Hypothesis: RSI divergence on 4h combined with 1d trend and volume confirmation provides high-quality signals.
# Bullish divergence (price makes lower low, RSI makes higher low) + 1d uptrend + volume spike = long.
# Bearish divergence (price makes higher high, RSI makes lower high) + 1d downtrend + volume spike = short.
# Uses 1d EMA for trend filter and volume confirmation to reduce false signals.
# Designed for low trade frequency (target: 20-40 trades/year) with high win rate.

name = "4h_RSI_Divergence_With_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need enough data for RSI (14), EMA (34), volume MA (20)
    start_idx = max(14, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # RSI divergence detection (need at least 3 bars back)
        if i >= 2:
            # Bullish divergence: price makes lower low, RSI makes higher low
            bullish_div = (close[i] < close[i-2]) and (rsi[i] > rsi[i-2])
            # Bearish divergence: price makes higher high, RSI makes lower high
            bearish_div = (close[i] > close[i-2]) and (rsi[i] < rsi[i-2])
        else:
            bullish_div = False
            bearish_div = False
        
        if position == 0:
            # Long entry: bullish divergence + daily uptrend + volume spike
            if bullish_div and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish divergence + daily downtrend + volume spike
            elif bearish_div and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 4h EMA(20) or divergence fails
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            if close[i] < ema_20 or not bullish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 4h EMA(20) or divergence fails
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
            if close[i] > ema_20 or not bearish_div:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals