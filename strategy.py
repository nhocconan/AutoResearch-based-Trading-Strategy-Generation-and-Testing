#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume confirmation.
- Camarilla H4/L4 levels from 1w provide strong weekly pivot points for 1d breakouts.
- 1w EMA50 trend filter ensures alignment with weekly momentum (works in bull/bear via trend alignment).
- Volume spike (>2.0x 20-period average) confirms breakout validity with higher threshold to reduce whipsaws.
- Discrete position sizing (0.25) balances return and drawdown control.
- Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
- Uses 1w HTF data loaded ONCE before loop per MTF rules.
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
    
    # Get 1w data ONCE before loop for Camarilla levels and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar (H, L, C)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla H4, L4 levels: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    camarilla_h4 = close_1w + (high_1w - low_1w) * 1.1 / 2
    camarilla_l4 = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (using previous completed 1w bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 with volume spike and above 1w EMA50
            if close[i] > camarilla_h4_aligned[i] and volume_spike[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L4 with volume spike and below 1w EMA50
            elif close[i] < camarilla_l4_aligned[i] and volume_spike[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla L4 OR below 1w EMA50
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla H4 OR above 1w EMA50
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H4L4_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0