#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend filter.
- Camarilla levels (H3, L3, H4, L4) calculated from prior 1d bar.
- Long when price breaks above H3 with close > H3 and volume spike, in uptrend (1w EMA34 rising).
- Short when price breaks below L3 with close < L3 and volume spike, in downtrend (1w EMA34 falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying H3 breakouts in uptrend, in bear via selling L3 breakdowns in downtrend.
- Uses 1w EMA34 for smooth trend filter that adapts to long-term market regime.
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
    
    # Get 1d data for Camarilla levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low)
    # L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    camarilla_H3_1d = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_L3_1d = close_1d - 1.0 * (high_1d - low_1d)
    camarilla_H4_1d = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_L4_1d = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h (1d -> 6h: 4x multiplier)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3_1d)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3_1d)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4_1d)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA34 trend
            if i > 0 and not np.isnan(ema_34_1w_aligned[i-1]):
                ema34_slope = ema_34_1w_aligned[i] - ema_34_1w_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    # Long: price breaks above H3 with close > H3 and volume spike
                    if close[i] > camarilla_H3_aligned[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    # Short: price breaks below L3 with close < L3 and volume spike
                    if close[i] < camarilla_L3_aligned[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 (reversal) or opposite signal
            if close[i] < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 (reversal) or opposite signal
            if close[i] > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0