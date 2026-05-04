#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1w trend filter and volume confirmation
# Long when price breaks above Alligator's Jaw AND 1w close > 1w EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below Alligator's Lips AND 1w close < 1w EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Uses 12h for precise entry timing, 1w for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Alligator: Jaw=EMA13(8), Teeth=EMA8(5), Lips=EMA5(3) - smoothed with future shift
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_WilliamsAlligator_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 12h timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Calculate 12h Williams Alligator
    # Jaw: 13-period EMA smoothed by 8 periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: 8-period EMA smoothed by 5 periods
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: 5-period EMA smoothed by 3 periods
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Jaw AND 1w uptrend AND volume spike
            if (close[i] > jaw[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Lips AND 1w downtrend AND volume spike
            elif (close[i] < lips[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Teeth OR 1w trend changes to downtrend
            if (close[i] < teeth[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Teeth OR 1w trend changes to uptrend
            if (close[i] > teeth[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals