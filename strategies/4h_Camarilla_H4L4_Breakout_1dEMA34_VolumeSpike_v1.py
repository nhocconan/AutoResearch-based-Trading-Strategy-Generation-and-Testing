#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H4, L4) from 1d timeframe as stronger support/resistance than H3/L3.
- Breakout above H4 with volume > 2.0x 20-bar average = long signal (more stringent than H3).
- Breakdown below L4 with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 1d EMA34 to align with daily trend (more robust than 12h for 4h).
- Designed for 4h timeframe to capture medium-term trends with fewer, higher-quality trades.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 19-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
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
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2  # H4 level
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2  # L4 level
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above H4 AND above 1d EMA34
                if close[i] > h4_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L4 AND below 1d EMA34
                elif close[i] < l4_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below L4 OR below 1d EMA34
            if close[i] < l4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above H4 OR above 1d EMA34
            if close[i] > h4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0