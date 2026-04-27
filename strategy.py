#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Price Action + 12h Trend + Volume Spike
# Uses 6h price crossing above/below 12h EMA50 as entry signal with volume confirmation.
# Works in bull markets (price above rising EMA) and bear markets (price below falling EMA).
# Target: 20-40 trades/year to minimize fee drag while capturing trend momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h close
    ema_50_12h = np.full(len(df_12h), np.nan)
    if len(close_12h) >= 50:
        # Use pandas EMA for accuracy and proper NaN handling
        ema_series = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean()
        ema_50_12h = ema_series.values
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA50 slope
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_12h_aligned[i-1]):
            ema_now = ema_50_12h_aligned[i]
            ema_prev = ema_50_12h_aligned[i-1]
            trend_up = ema_now > ema_prev
            trend_down = ema_now < ema_prev
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price crosses above 12h EMA50 + uptrend + volume spike
            if (close[i] > ema_50_12h_aligned[i] and 
                close[i-1] <= ema_50_12h_aligned[i-1] and  # crossed above
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below 12h EMA50 + downtrend + volume spike
            elif (close[i] < ema_50_12h_aligned[i] and 
                  close[i-1] >= ema_50_12h_aligned[i-1] and  # crossed below
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses back below 12h EMA50 or trend turns down
            if (close[i] < ema_50_12h_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above 12h EMA50 or trend turns up
            if (close[i] > ema_50_12h_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_PriceAction_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0