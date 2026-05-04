#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for higher timeframe trend alignment (works in bull/bear markets)
# Williams %R(14) identifies overbought/oversold conditions for mean reversion
# Volume confirmation (>1.3x 20 EMA) filters low-participation false signals
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "4h_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R(14) on 4h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) + uptrend + volume spike
            if williams_r[i] < -80 and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) + downtrend + volume spike
            elif williams_r[i] > -20 and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral OR trend changes OR weak volume
            if (williams_r[i] > -50 or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral OR trend changes OR weak volume
            if (williams_r[i] < -50 or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals