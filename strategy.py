#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume confirmation.
- 1h timeframe targets 60-150 total trades over 4 years (15-37/year).
- Uses 4h Camarilla levels from prior completed bar for swing points.
- 4h EMA34 ensures alignment with intermediate-term trend.
- Volume spike (>1.8x 20-period average) confirms breakout validity.
- Session filter (08-20 UTC) reduces noise during low-liquidity hours.
- Discrete position sizing (0.20) balances return and drawdown control.
- Works in bull/bear via trend filter and breakout logic.
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
    
    # Get 4h data ONCE before loop for Camarilla levels and EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar (H, L, C)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla H3, L3 levels: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_l3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (using previous completed 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # 4h EMA34 trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    # Session filter: 08-20 UTC (pre-compute hours from DatetimeIndex)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 with volume spike and above 4h EMA34
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla L3 with volume spike and below 4h EMA34
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and close[i] < ema_34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L3 OR below 4h EMA34
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above Camarilla H3 OR above 4h EMA34
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0