#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) + 4h EMA(50) trend + volume confirmation.
# Uses 4h EMA50 for trend direction (aligns with higher timeframe bias).
# Enters long when RSI < 30 (oversold) and price > 4h EMA50 (uptrend).
# Enters short when RSI > 70 (overbought) and price < 4h EMA50 (downtrend).
# Volume filter (>1.5x 20-period average) ensures institutional participation.
# Timeframe = 1h for precise entry timing; 4h for signal direction.
# Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (buys dips in uptrend) and bear markets (sells rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4-hour EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: RSI < 30 (oversold) and price > 4h EMA50 (uptrend) and volume
        if (rsi[i] < 30 and close[i] > ema50_4h_aligned[i] and volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short condition: RSI > 70 (overbought) and price < 4h EMA50 (downtrend) and volume
        elif (rsi[i] > 70 and close[i] < ema50_4h_aligned[i] and volume_filter[i]):
            signals[i] = -0.20
            position = -1
        # Exit conditions: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] >= 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] <= 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeFilter"
timeframe = "1h"
leverage = 1.0