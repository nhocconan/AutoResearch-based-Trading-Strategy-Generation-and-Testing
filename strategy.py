#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray power filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND Bull Power > 0 AND volume > 2x 20-period MA.
Short when Williams %R > -20 (overbought) AND Bear Power < 0 AND volume > 2x 20-period MA.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d HTF for Elder Ray trend filter to avoid counter-trend trades in strong moves,
volume spike for momentum confirmation, Williams %R for mean reversion in extremes.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R identifies overextended moves, Elder Ray confirms underlying bull/bear power,
volume spike adds conviction to reversals. Works in both bull (buy oversold dips) and bear (sell overbought rallies).
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
    
    # Calculate 6h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 1d Elder Ray power (Bull/Bear) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 20)  # Williams %R, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 2x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND Bull Power > 0 AND volume filter
            if wr < -80 and bull_power > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND Bear Power < 0 AND volume filter
            elif wr > -20 and bear_power < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if i >= start_idx + 1 and williams_r[i-1] <= -50 and wr > -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if i >= start_idx + 1 and williams_r[i-1] >= -50 and wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Power_VolumeSpike"
timeframe = "6h"
leverage = 1.0