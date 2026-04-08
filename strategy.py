#!/usr/bin/env python3
"""
1d Supertrend with 1-week Trend Filter and Volume Confirmation
Hypothesis: Supertrend identifies trend direction, while 1-week EMA filter avoids counter-trend trades.
Volume confirmation ensures institutional participation. Designed for 15-25 trades/year to minimize fee drag.
Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_supertrend_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Supertrend calculation (10, 3.0)
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[:atr_period+1])  # Seed with SMA
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            trend[i] = 1
        elif close[i] < supertrend[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1 and trend[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif trend[i] == -1 and trend[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif trend[i] == 1:
            supertrend[i] = max(supertrend[i-1], lower_band[i])
        else:
            supertrend[i] = min(supertrend[i-1], upper_band[i])
    
    # Volume filter: current volume > 2.0x 20-period average (stricter for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(supertrend[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Supertrend turns down OR price closes below Supertrend
            if trend[i] == -1 or close[i] < supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Supertrend turns up OR price closes above Supertrend
            if trend[i] == 1 or close[i] > supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1-week EMA50
            uptrend = close[i] > ema_50_1w_aligned[i]
            downtrend = close[i] < ema_50_1w_aligned[i]
            
            # Long: Supertrend uptrend + uptrend on 1w EMA + volume spike
            if (trend[i] == 1 and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Supertrend downtrend + downtrend on 1w EMA + volume spike
            elif (trend[i] == -1 and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals