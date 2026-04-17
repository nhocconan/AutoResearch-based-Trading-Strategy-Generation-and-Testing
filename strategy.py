#!/usr/bin/env python3
"""
12h_Stochastic_Ichimoku_Bounce_v1
Stochastic RSI + Ichimoku Cloud Bounce on 12h timeframe.
Uses daily Ichimoku cloud color for trend alignment (bullish/bearish cloud),
Stochastic RSI for mean reversion entries at extreme levels,
and volume confirmation for entry quality.
Designed to work in both bull and bear markets by combining trend following
with mean reversion at extreme Stochastic RSI levels during pullbacks.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === 1d Ichimoku Cloud ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 8:
            period9_high[i] = np.max(high_1d[i-8:i+1])
            period9_low[i] = np.min(low_1d[i-8:i+1])
        elif i > 0:
            start = max(0, i-8)
            period9_high[i] = np.max(high_1d[start:i+1])
            period9_low[i] = np.min(low_1d[start:i+1])
        else:
            period9_high[i] = high_1d[0]
            period9_low[i] = low_1d[0]
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 25:
            period26_high[i] = np.max(high_1d[i-25:i+1])
            period26_low[i] = np.min(low_1d[i-25:i+1])
        elif i > 0:
            start = max(0, i-25)
            period26_high[i] = np.max(high_1d[start:i+1])
            period26_low[i] = np.min(low_1d[start:i+1])
        else:
            period26_high[i] = high_1d[0]
            period26_low[i] = low_1d[0]
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 51:
            period52_high[i] = np.max(high_1d[i-51:i+1])
            period52_low[i] = np.min(low_1d[i-51:i+1])
        elif i > 0:
            start = max(0, i-51)
            period52_high[i] = np.max(high_1d[start:i+1])
            period52_low[i] = np.min(low_1d[start:i+1])
        else:
            period52_high[i] = high_1d[0]
            period52_low[i] = low_1d[0]
    senkou_b = (period52_high + period52_low) / 2
    
    # Bullish cloud: Senkou Span A > Senkou Span B
    bullish_cloud = senkou_a > senkou_b
    
    # === 12h Stochastic RSI (14,14,3,3) ===
    # RSI first
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    rsi_high = np.full_like(rsi, np.nan)
    rsi_low = np.full_like(rsi, np.nan)
    for i in range(len(rsi)):
        if i >= 14:
            rsi_high[i] = np.max(rsi[i-14:i+1])
            rsi_low[i] = np.min(rsi[i-14:i+1])
        elif i > 0:
            start = max(0, i-14)
            rsi_high[i] = np.max(rsi[start:i+1])
            rsi_low[i] = np.min(rsi[start:i+1])
        else:
            rsi_high[i] = rsi[0]
            rsi_low[i] = rsi[0]
    
    stoch_rsi = np.where((rsi_high - rsi_low) != 0, (rsi - rsi_low) / (rsi_high - rsi_low) * 100, 50)
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    # === Align 1d Ichimoku cloud to 12h timeframe ===
    bullish_cloud_aligned = align_htf_to_ltf(prices, df_1d, bullish_cloud.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_cloud_aligned[i]) or 
            np.isnan(stoch_rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish cloud (bullish trend) AND Stochastic RSI oversold (<20) AND volume confirmation
            if (bullish_cloud_aligned[i] > 0.5 and 
                stoch_rsi[i] < 20 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish cloud (bearish trend) AND Stochastic RSI overbought (>80) AND volume confirmation
            elif (bullish_cloud_aligned[i] <= 0.5 and 
                  stoch_rsi[i] > 80 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Stochastic RSI overbought (>80) OR cloud turns bearish
            if (stoch_rsi[i] > 80 or 
                bullish_cloud_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Stochastic RSI oversold (<20) OR cloud turns bullish
            if (stoch_rsi[i] < 20 or 
                bullish_cloud_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Stochastic_Ichimoku_Bounce_v1"
timeframe = "12h"
leverage = 1.0