#!/usr/bin/env python3
# 1d_RSI_1wTrend_VolumeSpike
# Hypothesis: Uses RSI(14) overbought/oversold on daily timeframe with weekly EMA(34) trend filter and volume spike confirmation.
# Weekly trend (1w EMA34) filters direction to avoid counter-trend trades. Volume > 2.0x 20-period MA confirms momentum.
# RSI < 30 for long, RSI > 70 for short in trending markets. Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in bull/bear by aligning with weekly trend. Position size 0.25 for balanced risk management.

name = "1d_RSI_1wTrend_VolumeSpike"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly EMA for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 34, 20)  # Warmup for RSI, weekly EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long entry: RSI oversold (<30) with volume confirmation, weekly uptrend
            if rsi[i] < 30 and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought (>70) with volume confirmation, weekly downtrend
            elif rsi[i] > 70 and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns above 50 or weekly trend turns down
            if rsi[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns below 50 or weekly trend turns up
            if rsi[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals