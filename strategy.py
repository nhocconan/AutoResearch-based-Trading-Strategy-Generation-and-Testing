#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 1d EMA34 Trend Filter and Volume Confirmation.
- Camarilla H3/L3 levels from 1d chart act as key intraday support/resistance; breakouts capture momentum with higher probability.
- 1d EMA34 provides higher-timeframe trend filter to align with dominant trend and reduce counter-trend trades.
- Volume confirmation (>1.5x 20-period average) validates breakout strength and reduces false signals.
- Discrete position sizing (0.25) minimizes fee churn while allowing meaningful returns.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
- Works in bull/bear markets via 1d trend filter and volatility-based volume confirmation.
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla H3 and L3 levels
        camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
        camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
        
        # Align Camarilla levels to 4h timeframe (using previous completed 1d bar)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # Volume confirmation: > 1.5x 20-period average volume (4h * 5 = ~20h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 with volume confirmation and above 1d EMA34 (bullish higher-timeframe trend)
            if close[i] > camarilla_h3_aligned[i] and volume_confirm[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with volume confirmation and below 1d EMA34 (bearish higher-timeframe trend)
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR below 1d EMA34 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR above 1d EMA34 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0