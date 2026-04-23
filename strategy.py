#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d Elder Ray regime filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 AND Bull Power > 0 AND 12h volume > 1.8x 20-period MA.
Short when Williams %R(14) crosses below -20 AND Bear Power < 0 AND 12h volume > 1.8x 20-period MA.
Exit when Williams %R crosses above -20 (for long) or below -80 (for short) or regime reverses.
Uses 1d HTF for Elder Ray regime filter to avoid counter-trend trades, volume spike for momentum confirmation.
Williams %R is effective in ranging markets (2025-2026 bearish/range) and catches reversals in bear rallies.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 1d Elder Ray for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 13, 20)  # Williams %R, EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R previous value for crossover
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            wr_cross_up = wr_prev <= -80 and wr > -80
            wr_cross_down = wr_prev >= -20 and wr < -20
        else:
            wr_cross_up = False
            wr_cross_down = False
        
        # Volume filter: 12h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        # Regime filters
        bull_regime = bull_power_val > 0
        bear_regime = bear_power_val < 0
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND bull regime AND volume filter
            if wr_cross_up and bull_regime and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND bear regime AND volume filter
            elif wr_cross_down and bear_regime and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -20 OR bull regime ends
                if wr >= -20 or bull_power_val <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -80 OR bear regime ends
                if wr <= -80 or bear_power_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Reversal_1dElderRay_Regime_VolumeSpike"
timeframe = "12h"
leverage = 1.0