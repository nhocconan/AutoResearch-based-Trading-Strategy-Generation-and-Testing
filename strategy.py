#!/usr/bin/env python3
# 6h_elder_ray_ema_rsi_v1
# Hypothesis: Combines Elder Ray (bull/bear power) with EMA trend and RSI filter for 6h timeframe.
# Long when Bull Power > 0, price > EMA20, RSI > 50; Short when Bear Power < 0, price < EMA20, RSI < 50.
# Uses 12h EMA50 trend filter to avoid counter-trend trades. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_ema_rsi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    ema13_period = 13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=ema13_period, adjust=False, min_periods=ema13_period).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # EMA20 for trend filter
    ema20_period = 20
    ema20 = close_series.ewm(span=ema20_period, adjust=False, min_periods=ema20_period).mean().values
    
    # RSI(14) for momentum filter
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full(n, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 12h data for trend filter (EMA50 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope: positive if current EMA > EMA 3 periods ago
    ema50_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(3, len(close_12h)):
        if not np.isnan(ema50_12h[i]) and not np.isnan(ema50_12h[i-3]):
            ema50_slope_12h[i] = ema50_12h[i] - ema50_12h[i-3]
    ema50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(ema13_period, ema20_period, rsi_period+1, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(ema20[i]) or np.isnan(rsi[i]) or np.isnan(ema50_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power < 0 or price < EMA20 or RSI < 40
            if bear_power[i] < 0 or close[i] < ema20[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 or price > EMA20 or RSI > 60
            if bull_power[i] > 0 or close[i] > ema20[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Bull Power > 0, price > EMA20, RSI > 50, 12h EMA50 slope positive
            if (bull_power[i] > 0 and 
                close[i] > ema20[i] and 
                rsi[i] > 50 and 
                ema50_slope_12h_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0, price < EMA20, RSI < 50, 12h EMA50 slope negative
            elif (bear_power[i] < 0 and 
                  close[i] < ema20[i] and 
                  rsi[i] < 50 and 
                  ema50_slope_12h_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals