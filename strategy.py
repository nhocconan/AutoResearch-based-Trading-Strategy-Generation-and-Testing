#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3L3 breakout with 12h EMA50 trend filter and volume spike.
- Camarilla H3/L3 levels act as intraday support/resistance from prior 12h period
- Breakout above H3 with volume > 2x average and price above 12h EMA50 → long
- Breakdown below L3 with volume > 2x average and price below 12h EMA50 → short
- 12h EMA50 filters counter-trend trades in bear markets
- Position size: 0.25 discrete level to minimize fee churn
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Works in bull via breakouts, in bear via 12h trend filter avoiding false breakouts
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
    
    # Calculate Camarilla levels from prior 12h period
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Use prior completed 12h bar for Camarilla calculation
    high_12h = df_12h['high'].shift(1).values  # Prior 12h high
    low_12h = df_12h['low'].shift(1).values    # Prior 12h low
    close_12h = df_12h['close'].shift(1).values # Prior 12h close
    
    # Camarilla formula: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + range_12h * 1.1 / 4
    camarilla_l3 = close_12h - range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for 12h bar close)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, 12h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 12h EMA50 AND volume confirmation
            if close[i] > h3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 12h EMA50 AND volume confirmation
            elif close[i] < l3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 12h EMA50
            if close[i] < l3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 12h EMA50
            if close[i] > h3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0