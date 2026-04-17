#!/usr/bin/env python3
"""
Hypothesis: Price action tends to respect key psychological levels derived from prior day's range.
Combining daily Camarilla pivot levels (R1/S1) with 4-hour volume spikes and trend alignment 
(4h EMA34) creates high-probability breakout trades. The strategy targets 20-30 trades per year 
by requiring confluence of three conditions: price breaking R1/S1, volume > 2x 20-bar average, 
and price on correct side of daily EMA34. Exits occur when price returns to the daily pivot 
or volume dries up, limiting adverse exposure in ranging markets. Designed for 4h timeframe 
to work in both bull (breakouts continuation) and bear (mean reversion to pivot) regimes.
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
    
    # Get daily data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla levels
    R1 = pivot + (range_ * 1.1 / 12)
    S1 = pivot - (range_ * 1.1 / 12)
    R2 = pivot + (range_ * 1.1 / 6)
    S2 = pivot - (range_ * 1.1 / 6)
    R3 = pivot + (range_ * 1.1 / 4)
    S3 = pivot - (range_ * 1.1 / 4)
    R4 = pivot + (range_ * 1.1 / 2)
    S4 = pivot - (range_ * 1.1 / 2)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(pivot_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and above daily EMA34
            if price > R1_4h[i] and vol > 2.0 * vol_ma and price > ema_34_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and below daily EMA34
            elif price < S1_4h[i] and vol > 2.0 * vol_ma and price < ema_34_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot or volume drops below average
            if price < pivot_4h[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot or volume drops below average
            if price > pivot_4h[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_EMA34"
timeframe = "4h"
leverage = 1.0