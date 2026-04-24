#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA34 trend filter and volume confirmation.
- Uses Camarilla pivot levels (H4, L4) from 1w timeframe as key support/resistance for 12h.
- Breakout above H4 with volume > 1.8x 24-bar average = long signal.
- Breakdown below L4 with volume > 1.8x 24-bar average = short signal.
- Trend filter: price must be above/below 1w EMA34 to align with weekly direction.
- Designed for 12h timeframe to capture swing trades with proper weekly structure.
- Uses discrete position size 0.28 to balance return and drawdown control.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Combines proven Camarilla structure with weekly trend filter for BTC/ETH resilience.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1w timeframe
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    
    # Align Camarilla levels to 12h timeframe (wait for 1w bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.8x 24-period average (2x 12h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout
            if volume_confirm:
                # Long: price breaks above H4 AND above 1w EMA34
                if close[i] > h4_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.28
                    position = 1
                # Short: price breaks below L4 AND below 1w EMA34
                elif close[i] < l4_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.28
                    position = -1
        elif position == 1:
            # Long exit: price crosses below L4 OR below 1w EMA34
            if close[i] < l4_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Short exit: price crosses above H4 OR above 1w EMA34
            if close[i] > h4_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0