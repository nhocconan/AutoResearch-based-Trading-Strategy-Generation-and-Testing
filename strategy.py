#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for trend direction and 1h for entry timing.
# Long when 4h EMA21 > 1d EMA50, 1h RSI < 30 (oversold), volume > 1.3x average.
# Short when 4h EMA21 < 1d EMA50, 1h RSI > 70 (overbought), volume > 1.3x average.
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce trade frequency.
# Session filter (08-20 UTC) to avoid low-volume periods.
# Fixed position size 0.20 to limit drawdown and control trade frequency.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.

name = "1h_4h1dEMA_RSI_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h EMA21
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d EMA50
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h EMA21 > 1d EMA50 (uptrend), 1h RSI < 30 (oversold), volume spike
            if (ema_4h_aligned[i] > ema_1d_aligned[i] and
                rsi[i] < 30 and
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA21 < 1d EMA50 (downtrend), 1h RSI > 70 (overbought), volume spike
            elif (ema_4h_aligned[i] < ema_1d_aligned[i] and
                  rsi[i] > 70 and
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend reversal or RSI > 50 (mean reversion)
            if (ema_4h_aligned[i] <= ema_1d_aligned[i] or
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend reversal or RSI < 50 (mean reversion)
            if (ema_4h_aligned[i] >= ema_1d_aligned[i] or
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals