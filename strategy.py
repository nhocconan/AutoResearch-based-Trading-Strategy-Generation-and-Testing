#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation. Enters long when price breaks above R3 with volume > 1.5x 20-period average and 1d EMA34 uptrend. Enters short when price breaks below S3 with volume > 1.5x 20-period average and 1d EMA34 downtrend. Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-37/year) to avoid fee drag while capturing strong momentum moves in both bull and bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for volatility filtering
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate Camarilla levels from previous 12h bar
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(1, n):
        camarilla_r3[i] = close[i-1] + 1.1 * (high[i-1] - low[i-1]) / 2
        camarilla_s3[i] = close[i-1] - 1.1 * (high[i-1] - low[i-1]) / 2
    # First bar: use same values
    camarilla_r3[0] = camarilla_r3[1] if n > 1 else close[0]
    camarilla_s3[0] = camarilla_s3[1] if n > 1 else close[0]
    
    # Volume spike: volume > 1.5x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    # 1d EMA34 for HTF trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume EMA, 34 for 1d EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(vol_ema_20[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long entry: price breaks above R3 + volume spike + 1d uptrend
        if close[i] > camarilla_r3[i] and volume_spike[i] and htf_trend[i] == 1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short entry: price breaks below S3 + volume spike + 1d downtrend
        elif close[i] < camarilla_s3[i] and volume_spike[i] and htf_trend[i] == -1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: reverse signal or loss of trend alignment
        elif position == 1 and (close[i] < camarilla_s3[i] or htf_trend[i] == -1):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_r3[i] or htf_trend[i] == 1):
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0