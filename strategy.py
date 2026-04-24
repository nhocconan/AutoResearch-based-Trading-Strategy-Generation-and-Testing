#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend filter.
- Camarilla levels: calculated from prior 12h bar's high/low/close.
  H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4.
  Long when price breaks above H3 with volume spike, short when breaks below L3.
- Trend filter: Only trade in direction of 12h EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakouts in downtrend.
- Camarilla levels provide institutional support/resistance, EMA34 filters counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Calculate Camarilla levels from prior 12h bar
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3_12h = close_12h + 1.1 * (high_12h - low_12h) / 4
    camarilla_l3_12h = close_12h - 1.1 * (high_12h - low_12h) / 4
    
    # Align Camarilla levels to 6h (they represent levels for the completed 12h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 12h EMA34 trend
            if i > 0 and not np.isnan(ema_34_12h_aligned[i-1]):
                ema34_slope = ema_34_12h_aligned[i] - ema_34_12h_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long when price breaks above H3 with volume spike
                    if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short when price breaks below L3 with volume spike
                    if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 (reversal) or opposite signal
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 (reversal) or opposite signal
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0