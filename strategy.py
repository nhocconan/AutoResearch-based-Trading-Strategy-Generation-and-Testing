#!/usr/bin/env python3
"""
6h_1d_Camarilla_PriceAction
Hypothesis: Trade reversals at Camarilla pivot levels (R3/S3, R4/S4) on 6h timeframe with 1d trend filter and volume confirmation.
Camarilla levels from daily timeframe act as intraday support/resistance. Price rejection at these levels with volume divergence
indicates exhaustion. 1d EMA50 ensures trades align with intermediate trend to avoid counter-trend whipsaws.
Target: 15-25 trades/year to minimize fee drag while capturing high-probability reversals.
Works in both bull/bear markets as reversals occur at all market phases.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Camarilla: H-L range, levels at specific fractions
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_S4 = np.zeros(len(df_1d))
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_R4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_S3[i] = camarilla_S4[i] = camarilla_R3[i] = camarilla_R4[i] = np.nan
        else:
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            range_val = H - L
            camarilla_S4[i] = C - ((H - L) * 1.1 / 2)
            camarilla_S3[i] = C - ((H - L) * 1.1 / 4)
            camarilla_R3[i] = C + ((H - L) * 1.1 / 4)
            camarilla_R4[i] = C + ((H - L) * 1.1 / 2)
    
    # Align Camarilla levels to 6h
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Get daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume < 0.5x 20-period average (low volume on rejection)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    low_volume = volume < (vol_ma_20 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(low_volume[i])):
            signals[i] = 0.0
            continue
        
        # Long: rejection at S3/S4 with low volume and price above daily EMA50
        long_reject_S3 = (low[i] <= S3_aligned[i] * 1.001) and (close[i] > S3_aligned[i])  # touched S3, closed above
        long_reject_S4 = (low[i] <= S4_aligned[i] * 1.001) and (close[i] > S4_aligned[i])  # touched S4, closed above
        long_condition = (long_reject_S3 or long_reject_S4) and low_volume[i] and (close[i] > ema_50_aligned[i])
        
        # Short: rejection at R3/R4 with low volume and price below daily EMA50
        short_reject_R3 = (high[i] >= R3_aligned[i] * 0.999) and (close[i] < R3_aligned[i])  # touched R3, closed below
        short_reject_R4 = (high[i] >= R4_aligned[i] * 0.999) and (close[i] < R4_aligned[i])  # touched R4, closed below
        short_condition = (short_reject_R3 or short_reject_R4) and low_volume[i] and (close[i] < ema_50_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Camarilla_PriceAction"
timeframe = "6h"
leverage = 1.0