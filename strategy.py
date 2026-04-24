#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Camarilla levels.
- Camarilla pivot levels (H3, L3) from prior 1d: Long when price > H3, Short when price < L3.
- Trend filter: Only trade in direction of 1d EMA34 (long if EMA34 rising, short if falling).
- Volume confirmation: current volume > 2.0x 30-period volume MA to ensure strong participation.
- Discrete signal size: 0.30 to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses actual Camarilla calculation: H3 = H + 1.1*(L-C)/4, L3 = L - 1.1*(H-C)/4.
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla H3 and L3 levels from prior 1d bar
    # H3 = H + 1.1*(L - C)/4, L3 = L - 1.1*(H - C)/4
    camarilla_H3 = high_1d + 1.1 * (low_1d - close_1d) / 4.0
    camarilla_L3 = low_1d - 1.1 * (high_1d - close_1d) / 4.0
    
    # Align to 12h: use prior 1d's levels (already completed bar)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 30-period volume MA
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA34 trend
            if i > 0 and not np.isnan(ema_34_1d_aligned[i-1]):
                ema34_slope = ema_34_1d_aligned[i] - ema_34_1d_aligned[i-1]
                if ema34_slope > 0:  # Uptrend
                    if close[i] > camarilla_H3_aligned[i] and volume_spike[i]:
                        # Buy on H3 breakout in uptrend
                        signals[i] = 0.30
                        position = 1
                elif ema34_slope < 0:  # Downtrend
                    if close[i] < camarilla_L3_aligned[i] and volume_spike[i]:
                        # Sell on L3 breakdown in downtrend
                        signals[i] = -0.30
                        position = -1
        elif position == 1:
            # Long exit: price returns to Camarilla L3 level or opposite break
            if not np.isnan(camarilla_L3_aligned[i]) and not np.isnan(camarilla_H3_aligned[i]):
                if close[i] < camarilla_L3_aligned[i] or close[i] < camarilla_H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
        elif position == -1:
            # Short exit: price returns to Camarilla H3 level or opposite break
            if not np.isnan(camarilla_H3_aligned[i]) and not np.isnan(camarilla_L3_aligned[i]):
                if close[i] > camarilla_H3_aligned[i] or close[i] > camarilla_L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0