#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from 4h timeframe as strong support/resistance.
- Breakout above H3 with volume > 2.0x 20-bar average = long signal.
- Breakdown below L3 with volume > 2.0x 20-bar average = short signal.
- Trend filter: price must be above/below 4h EMA34 to align with 4h trend.
- Designed for 1h timeframe to capture swings with higher probability entries.
- Uses discrete position size 0.20 to limit drawdown and reduce fee churn.
- Targets 15-37 trades/year (60-150 total over 4 years) to stay fee-efficient.
- Volume confirmation reduces false breakouts in choppy markets.
- Session filter (08-20 UTC) to avoid low-liquidity periods.
- Novelty: Uses H3/L3 levels (stronger breakout levels than H4/L4) and 4h EMA34 on 1h timeframe.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data ONCE before loop for Camarilla levels and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 4h timeframe
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 4  # H3 level
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 4  # L3 level
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 4h EMA34 trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms breakout AND in session
            if volume_confirm and in_session:
                # Long: price breaks above H3 AND above 4h EMA34
                if close[i] > h3_aligned[i] and close[i] > ema_34_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below L3 AND below 4h EMA34
                elif close[i] < l3_aligned[i] and close[i] < ema_34_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price crosses below L3 OR below 4h EMA34 OR outside session
            if close[i] < l3_aligned[i] or close[i] < ema_34_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above H3 OR above 4h EMA34 OR outside session
            if close[i] > h3_aligned[i] or close[i] > ema_34_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0