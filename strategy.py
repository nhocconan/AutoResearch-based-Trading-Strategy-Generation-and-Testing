#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1D_Trend_Force
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries, confirmed by 4-hour volume spike (>1.5x 20-period average) and filtered by 1-day EMA trend direction. Enter long when price breaks above R1 with volume confirmation and price above daily EMA34, short when price breaks below S1 with volume confirmation and price below daily EMA34. Camarilla levels provide high-probability intraday support/resistance, and the daily trend filter ensures alignment with higher timeframe momentum. Designed for 4h timeframe to target 20-50 trades/year, avoiding excessive fee churn while capturing meaningful breakouts in both bull and bear markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1D_Trend_Force"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no extra delay for pivot points)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above daily EMA34
            if close[i] > r1_aligned[i] and vol_spike and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below daily EMA34
            elif close[i] < s1_aligned[i] and vol_spike and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or volume drops significantly
            if close[i] < s1_aligned[i] or volume[i] < 0.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or volume drops significantly
            if close[i] > r1_aligned[i] or volume[i] < 0.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals