#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_VolumeFilter
Hypothesis: Camarilla R1/S1 levels from 1-day timeframe act as strong intraday support/resistance.
Price breaking above R1 or below S1 with volume confirmation indicates institutional interest.
Works in bull/bear by taking breakouts in direction of 1-day trend (EMA34 filter).
Target: 12-37 trades/year on 12h timeframe. Uses discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC data"""
    typical = (high + low + close) / 3
    range_val = high - low
    # Camarilla levels
    R4 = close + range_val * 1.500
    R3 = close + range_val * 1.250
    R2 = close + range_val * 1.166
    R1 = close + range_val * 1.083
    S1 = close - range_val * 1.083
    S2 = close - range_val * 1.166
    S3 = close - range_val * 1.250
    S4 = close - range_val * 1.500
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on daily data
    R1_1d = np.full_like(close_1d, np.nan)
    S1_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        R1, R2, R3, R4, S1, S2, S3, S4 = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d[i]
        )
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    # 1-day EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D indicators to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if NaN in critical values
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].values[i]
        r1 = R1_1d_aligned[i]
        s1 = S1_1d_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 + volume + price above 1D EMA (uptrend)
            if price > r1 and vol_conf and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + price below 1D EMA (downtrend)
            elif price < s1 and vol_conf and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below S1 or loses volume confirmation
            if price < s1 or vol_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or loses volume confirmation
            if price > r1 or vol_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0