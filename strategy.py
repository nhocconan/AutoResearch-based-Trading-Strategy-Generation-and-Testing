#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_RSI_WeeklyTrend_VolumeSpike"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14) for trend filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs))
    rsi_14_6h = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Daily RSI(14) for entry signal
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
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_6h = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_6h[i]) or np.isnan(rsi_14_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: RSI > 50 = uptrend, RSI < 50 = downtrend
        weekly_uptrend = rsi_14_6h[i] > 50
        weekly_downtrend = rsi_14_6h[i] < 50
        
        if position == 0:
            # Long entry: daily RSI < 30 (oversold) + weekly uptrend + volume spike
            long_cond = (rsi_14_6h[i] < 30 and weekly_uptrend and vol_spike[i])
            
            # Short entry: daily RSI > 70 (overbought) + weekly downtrend + volume spike
            short_cond = (rsi_14_6h[i] > 70 and weekly_downtrend and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: daily RSI > 50 (mean reversion) or weekly trend turns down
            if rsi_14_6h[i] > 50 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: daily RSI < 50 (mean reversion) or weekly trend turns up
            if rsi_14_6h[i] < 50 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI mean reversion on 6h timeframe with weekly trend filter and volume spike confirmation.
# Enters long when daily RSI < 30 (oversold), weekly RSI > 50 (uptrend), and volume spike.
# Enters short when daily RSI > 70 (overbought), weekly RSI < 50 (downtrend), and volume spike.
# Exits when daily RSI crosses back above/below 50 or weekly trend changes.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Volume spike confirms institutional interest at extremes.
# Targets 15-35 trades/year on 6h timeframe with discrete sizing (0.25) to minimize churn.