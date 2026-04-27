# 12h_HybridTrendReversal_v1
# Hypothesis: Combines 1-day trend filtering (EMA50) with 12-hour mean reversion (RSI) and volume confirmation.
# The 1-day EMA establishes the higher timeframe bias, while RSI on 12h identifies overextended moves against the trend.
# Volume spike confirms institutional participation. Works in bull/bear by trading pullbacks in the trend.
# Target: 12-37 trades/year (50-150 total over 4 years).

#!/usr/bin/env python3
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-day EMA
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 12h data for RSI and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 14-period RSI on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume on 12h
    vol_avg_period = 20
    vol_avg_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= vol_avg_period:
        for i in range(vol_avg_period - 1, len(volume_12h)):
            vol_avg_12h[i] = np.mean(volume_12h[i - vol_avg_period + 1:i + 1])
    
    # Align indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA, RSI, and volume average
    start_idx = max(50, 14, 20)  # EMA50, RSI14, VolAvg20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_1d_aligned[i]
        rsi = rsi_12h_aligned[i]
        vol_avg = vol_avg_12h_aligned[i]
        vol_current = volume[i]
        
        # Volume spike: current volume > 1.5x average
        volume_spike = vol_current > 1.5 * vol_avg
        
        if position == 0:
            # Long: Uptrend (price above daily EMA) + oversold RSI + volume spike
            if (price > ema_trend and rsi < 30 and volume_spike):
                signals[i] = size
                position = 1
            # Short: Downtrend (price below daily EMA) + overbought RSI + volume spike
            elif (price < ema_trend and rsi > 70 and volume_spike):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or trend fails (price below daily EMA)
            if rsi > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or trend fails (price above daily EMA)
            if rsi < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_HybridTrendReversal_v1"
timeframe = "12h"
leverage = 1.0