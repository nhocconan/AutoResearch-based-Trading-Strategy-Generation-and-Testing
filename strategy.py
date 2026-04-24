#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 12h EMA34 rising AND volume > 1.5x 20-period average
- Short when price breaks below Camarilla L3 AND 12h EMA34 falling AND volume > 1.5x 20-period average
- Exit on opposite Camarilla break (L3 for long, H3 for short) or trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Targets 75-200 trades over 4 years by requiring confluence of breakout, trend, and volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous bar
    # H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    # Using previous bar's high/low/close to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # seed first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h EMA34 slope: rising if current > previous, falling if current < previous
    ema_rising = np.zeros_like(ema_34_12h_aligned, dtype=bool)
    ema_falling = np.zeros_like(ema_34_12h_aligned, dtype=bool)
    ema_rising[1:] = ema_34_12h_aligned[1:] > ema_34_12h_aligned[:-1]
    ema_falling[1:] = ema_34_12h_aligned[1:] < ema_34_12h_aligned[:-1]
    
    # Discrete position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND EMA34 rising AND volume spike
            if close[i] > camarilla_h3[i] and ema_rising[i] and volume_spike[i]:
                signals[i] = position_size
                position = 1
            # Short: break below L3 AND EMA34 falling AND volume spike
            elif close[i] < camarilla_l3[i] and ema_falling[i] and volume_spike[i]:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: break below L3 OR EMA34 starts falling
            if close[i] < camarilla_l3[i] or ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: break above H3 OR EMA34 starts rising
            if close[i] > camarilla_h3[i] or ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0