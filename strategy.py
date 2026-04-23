#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d EMA50 trend filter and volume confirmation.
- Bollinger Bands: 20-period SMA, 2 standard deviations
- Squeeze condition: BB Width < 20-period percentile 20 (low volatility)
- Breakout: Close > Upper Band (long) or Close < Lower Band (short)
- Trend filter: Price > 1d EMA50 for longs, Price < 1d EMA50 for shorts
- Volume confirmation: Volume > 1.5x 20-period average
- Exit: Opposite band touch or squeeze re-engagement
- Uses Bollinger Squeeze for low-volatility breakout anticipation, EMA50 for HTF direction
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in ranging markets (squeeze breakouts) and trending markets (continuation breakouts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # Normalized width
    
    # Bollinger Squeeze: BB Width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile_20 = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile_20
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 50)  # Need 50 for EMA50 and percentile, 20 for BB
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(bb_width_percentile_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Bollinger Squeeze condition
        is_squeeze = squeeze_condition[i]
        
        if position == 0:
            # Long: Squeeze breakout up + price > 1d EMA50 + volume confirmation
            if (is_squeeze and 
                close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze breakout down + price < 1d EMA50 + volume confirmation
            elif (is_squeeze and 
                  close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price touches lower band OR squeeze re-engagement
            if close[i] < lower_band[i] or not is_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price touches upper band OR squeeze re-engagement
            if close[i] > upper_band[i] or not is_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerSqueeze_Breakout_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0