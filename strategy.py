#!/usr/bin/env python3
# 1h_4h1d_VolumeBreakout_TrendFilter
# Hypothesis: Use 4h EMA50 trend filter and 1d RSI momentum filter to gate 1h volume breakout signals.
# Enter long when price breaks above 1h high of previous 20 bars with volume > 2x average, 4h uptrend, and 1d RSI > 50.
# Enter short when price breaks below 1h low of previous 20 bars with volume > 2x average, 4h downtrend, and 1d RSI < 50.
# Exit when trend reverses or opposite breakout occurs. Target: 15-30 trades/year to minimize fee drag on 1h timeframe.
# Works in bull markets via trend-following breaks and in bear markets via short breakdowns with trend alignment.

name = "1h_4h1d_VolumeBreakout_TrendFilter"
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
    
    # 4h trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d momentum filter (RSI14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[:13])
    avg_loss[13] = np.mean(loss[:13])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    rsi_50 = rsi > 50
    
    # Align 1d RSI to 1h
    rsi_50_aligned = align_htf_to_ltf(prices, df_1d, rsi_50.astype(float))
    
    # Volume breakout: 20-period high/low and 2x volume average
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    vol_ma_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    volume_breakout = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(rsi_50_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_breakout[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high with volume confirmation, 4h uptrend, and 1d RSI > 50
            if (high[i] > highest_20[i] and
                trend_4h_up_aligned[i] > 0.5 and
                rsi_50_aligned[i] > 0.5 and
                volume_breakout[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low with volume confirmation, 4h downtrend, and 1d RSI < 50
            elif (low[i] < lowest_20[i] and
                  trend_4h_down_aligned[i] > 0.5 and
                  rsi_50_aligned[i] < 0.5 and
                  volume_breakout[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 20-period low or 4h trend turns down
            if (low[i] < lowest_20[i] or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above 20-period high or 4h trend turns up
            if (high[i] > highest_20[i] or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals