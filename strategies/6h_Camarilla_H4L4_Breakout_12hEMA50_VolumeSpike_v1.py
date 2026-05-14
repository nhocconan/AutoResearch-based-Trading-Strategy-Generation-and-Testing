#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 Breakout with 12h EMA50 Trend Filter and Volume Spike.
- Camarilla H4/L4 levels from 12h chart act as key support/resistance; breakouts capture momentum with higher probability.
- 12h EMA50 provides higher-timeframe trend filter to align with intermediate momentum and reduce counter-trend trades.
- Volume spike (>2.0x 24-period average) confirms breakout validity and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 50-150 total over 4 years (12-37/year) on 6h timeframe to avoid fee drag.
- Works in bull/bear markets via 12h trend filter and volatility-based volume confirmation.
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
    
    # Get 12h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from 12h OHLC
    if len(df_12h) >= 2:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Camarilla H4 and L4 levels
        camarilla_h4 = close_12h + 1.1 * (high_12h - low_12h) / 2
        camarilla_l4 = close_12h - 1.1 * (high_12h - low_12h) / 2
        
        # Align Camarilla levels to 6h timeframe (using previous completed 12h bar)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # Volume confirmation: > 2.0x 24-period average volume (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H4 with volume spike and above 12h EMA50 (bullish higher-timeframe trend)
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L4 with volume spike and below 12h EMA50 (bearish higher-timeframe trend)
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below L4 OR below 12h EMA50 (trend change)
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H4 OR above 12h EMA50 (trend change)
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0