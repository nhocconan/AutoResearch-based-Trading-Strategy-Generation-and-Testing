#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout with 4h EMA50 Trend Filter and Volume Spike.
- Camarilla H3/L3 from 4h chart act as key support/resistance; breakouts capture momentum.
- 4h EMA50 provides higher-timeframe trend filter to align with intermediate momentum.
- Volume spike (>2.0x 24-period average) confirms breakout validity.
- Session filter (08-20 UTC) reduces noise trades.
- Discrete position sizing (0.20) minimizes fee churn.
- Target trades: 60-150 total over 4 years (15-37/year) on 1h timeframe.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from 4h OHLC
    if len(df_4h) >= 2:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        # Camarilla H3 and L3 levels
        camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 4
        camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 4
        
        # Align Camarilla levels to 1h timeframe (using previous completed 4h bar)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # Volume confirmation: > 2.0x 24-period average volume (1h * 24 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50) + 1
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]) or not in_session):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 with volume spike and above 4h EMA50 (bullish higher-timeframe trend)
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below L3 with volume spike and below 4h EMA50 (bearish higher-timeframe trend)
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR below 4h EMA50 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above H3 OR above 4h EMA50 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0