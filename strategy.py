#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1dTrendFilter
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with volume confirmation and 1d EMA(34) trend filter. 
Camarilla levels from daily range provide institutional support/resistance. Volume > 1.5x 12-period average confirms breakout strength.
1d EMA(34) ensures trades align with higher timeframe trend, reducing whipsaws in ranging markets.
Target: 12-30 trades/year to stay within 50-150 total trades over 4 years, minimizing fee drag.
Works in bull/bear by trading breakouts with trend alignment and volume confirmation.
"""

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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        camarilla_R1[i] = close_1d[i-1] + (range_ * 1.1 / 12)
        camarilla_S1[i] = close_1d[i-1] - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 1d EMA(34) trend filter
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 12-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 12
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period, ema_period)  # Start after we have all data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume and above daily EMA
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume and below daily EMA
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Camarilla S1 or below daily EMA
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Camarilla R1 or above daily EMA
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_Volume_1dTrendFilter"
timeframe = "12h"
leverage = 1.0