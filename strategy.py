#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA),
# close > 1d EMA50, and volume > 2.0x 20-bar avg.
# Short when jaws < teeth < lips, close < 1d EMA50, and volume > 2.0x 20-bar avg.
# Exit when Alligator lines cross (jaws < teeth for long, jaws > teeth for short).
# Uses 12h timeframe for low trade frequency (target: 12-37/year) to minimize fee drag.
# Williams Alligator identifies trending markets via SMMA alignment.
# 1d EMA50 filters for higher timeframe trend alignment.
# Volume confirmation reduces false signals.
# Works in bull markets via aligned Alligator + uptrend and in bear markets via aligned Alligator + downtrend.
# Target: 50-150 total trades over 4 years.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    # SMMA today = (SMMA yesterday * (period-1) + close today) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaws = smma(close, 13)  # 13-period SMMA (blue line)
    teeth = smma(close, 8)   # 8-period SMMA (red line)
    lips = smma(close, 5)    # 5-period SMMA (green line)
    
    # Volume confirmation: volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for SMMA and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaws = jaws[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: jaws > teeth > lips (Alligator aligned up), close > 1d EMA50, volume spike
            if (curr_jaws > curr_teeth and curr_teeth > curr_lips and
                curr_close > curr_ema_50_1d and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: jaws < teeth < lips (Alligator aligned down), close < 1d EMA50, volume spike
            elif (curr_jaws < curr_teeth and curr_teeth < curr_lips and
                  curr_close < curr_ema_50_1d and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Alligator lines cross (jaws < teeth)
            if curr_jaws < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Alligator lines cross (jaws > teeth)
            if curr_jaws > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals