#!/usr/bin/env python3
# 12h_RSI_Extremes_1dTrend_With_VolumeFilter
# Strategy: Trade RSI extremes (RSI<30 for long, RSI>70 for short) only when aligned with 1d trend (EMA50) and confirmed by volume spike (volume > 1.5x 20-period average)
# Exit when RSI returns to neutral zone (40-60 for longs, 40-60 for shorts)
# Designed for 12h timeframe with selective entries to minimize trade frequency and avoid whipsaws
# Uses volume confirmation to filter false signals and trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_RSI_Extremes_1dTrend_With_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smooth = wilders_smooth(gain, 14)
    loss_smooth = wilders_smooth(loss, 14)
    
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: RSI oversold, above 1d EMA50 (uptrend filter), and volume spike
            if rsi[i] < 30 and close[i] > ema_50_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought, below 1d EMA50 (downtrend filter), and volume spike
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral zone
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral zone
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals