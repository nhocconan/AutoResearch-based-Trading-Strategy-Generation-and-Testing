#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ATR Stop
Hypothesis: Weekly Donchian channels (20-day high/low) capture major trend breakouts.
Breakouts with volume confirmation capture institutional moves, while ATR-based stops limit losses.
Works in both bull and bear markets by using weekly levels and avoiding overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for Donchian and ATR
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly True Range for ATR
    tr1 = np.abs(high_weekly - low_weekly)
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr1[0] = high_weekly[0] - low_weekly[0]
    tr2[0] = np.abs(high_weekly[0] - close_weekly[0])
    tr3[0] = np.abs(low_weekly[0] - close_weekly[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    atr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_weekly)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Main timeframe data (1d)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_weekly_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_weekly_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 30-period average
        vol_ma = np.mean(volume[max(0, i-30):i]) if i >= 30 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume confirmation
            if price > upper and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below weekly Donchian low with volume confirmation
            elif price < lower and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly Donchian low or ATR-based stop
            if price < lower or (i > 0 and close[i-1] > lower and price < close[i-1] - 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly Donchian high or ATR-based stop
            if price > upper or (i > 0 and close[i-1] < upper and price > close[i-1] + 1.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0