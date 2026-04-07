#!/usr/bin/env python3
"""
6h_cci_pullback_12h_trend_volume_v1
Hypothesis: On 6h timeframe, enter long on CCI pullback below -50 when 12h EMA50 is rising and volume is above average; enter short on pullback above +50 when 12h EMA50 is falling and volume is above average. Uses CCI mean reversion within a trend filter to avoid whipsaw in both bull and bear markets. Target: 15-30 trades/year (~60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_pullback_12h_trend_volume_v1"
timeframe = "6h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50 = df_12h['close'].ewm(span=50, adjust=False).mean()
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50.values)
    
    # 12h EMA50 slope for trend direction (rising/falling)
    ema_50_slope = ema_50.diff()
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope.values)
    
    # CCI(20) on 6h: (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp.values) / (0.015 * mad.values)
    cci = np.nan_to_num(cci, nan=0.0)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(cci[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_50_slope_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: CCI rises above 0 or 12h EMA50 slope turns negative
            if cci[i] > 0 or ema_50_slope_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: CCI falls below 0 or 12h EMA50 slope turns positive
            if cci[i] < 0 or ema_50_slope_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: CCI < -50 (pullback in uptrend), with volume and rising 12h EMA50
            if (cci[i] < -50 and vol_confirm and 
                ema_50_slope_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: CCI > 50 (pullback in downtrend), with volume and falling 12h EMA50
            elif (cci[i] > 50 and vol_confirm and 
                  ema_50_slope_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals