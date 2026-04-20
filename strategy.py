#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_VolumeTrend_Regime
# Hypothesis: Camarilla pivot levels from 1-day timeframe combined with volume confirmation and 1-week trend filter (via EMA) to capture institutional breakouts in both bull and bear markets.
# Uses Camarilla R1/S1 levels for institutional entry points, volume for confirmation, and 1-week EMA for trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "12h_Camarilla_R1_S1_Breakout_VolumeTrend_Regime"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-week EMA for trend filter (using close)
    close_1w = get_htf_data(prices, '1w')['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), ema_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use the previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i > 0:  # Need previous day's data
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            camarilla_R1[i] = C + (H - L) * 1.1 / 12
            camarilla_S1[i] = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Ensure EMA and volume MA are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + above weekly EMA + volume confirmation
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 + below weekly EMA + volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Camarilla S1 or below weekly EMA
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Camarilla R1 or above weekly EMA
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals