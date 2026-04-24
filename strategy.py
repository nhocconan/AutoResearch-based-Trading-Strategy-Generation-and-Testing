#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend.
- Camarilla levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2 from prior day.
- Long when price breaks above H3 with volume confirmation in uptrend (1w EMA34 rising).
- Short when price breaks below L3 with volume confirmation in downtrend (1w EMA34 falling).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses EMA for smooth trend and volume spike for institutional participation confirmation.
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior day (1d timeframe)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_H3 = close + 1.1 * (high - low) / 2
    camarilla_L3 = close - 1.1 * (high - low) / 2
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where volume MA is ready
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout/breakdown with volume confirmation
            if camarilla_H3[i] > camarilla_H3[i-1]:  # Ensure level is valid
                # Uptrend: EMA34 rising
                if i > 0 and not np.isnan(ema_34_1w_aligned[i-1]):
                    ema34_slope = ema_34_1w_aligned[i] - ema_34_1w_aligned[i-1]
                    if ema34_slope > 0 and close[i] > camarilla_H3[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
            
            if camarilla_L3[i] < camarilla_L3[i-1]:  # Ensure level is valid
                # Downtrend: EMA34 falling
                if i > 0 and not np.isnan(ema_34_1w_aligned[i-1]):
                    ema34_slope = ema_34_1w_aligned[i] - ema_34_1w_aligned[i-1]
                    if ema34_slope < 0 and close[i] < camarilla_L3[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 or opposite signal
            if close[i] < camarilla_L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 or opposite signal
            if close[i] > camarilla_H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0