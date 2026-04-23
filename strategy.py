#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray power filter and volume confirmation.
Long when Williams %R crosses above -80 from below AND Bear Power < 0 (bearish momentum weakening) AND volume > 1.5x 20-period MA.
Short when Williams %R crosses below -20 from above AND Bull Power > 0 (bullish momentum weakening) AND volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) OR Elder Ray power reverses.
Uses 1d HTF for Elder Ray to filter counter-trend trades in ranging markets, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R identifies overextended conditions, Elder Ray measures underlying power, volume confirms reversal strength.
Works in both bull and bear markets by fading extremes when higher timeframe momentum is weakening.
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
    for i in range(14, n):
        highest_high = np.max(high[i-14:i+1])
        lowest_low = np.min(low[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # avoid division by zero
    
    # Calculate 1d Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13) (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 20)  # Williams %R (needs 14), Elder Ray (needs 13), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        # Williams %R cross conditions
        wr_cross_above_80 = wr_prev <= -80 and wr > -80  # crosses above -80 from below
        wr_cross_below_20 = wr_prev >= -20 and wr < -20  # crosses below -20 from above
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND Bear Power < 0 (weakening bearish momentum) AND volume filter
            if wr_cross_above_80 and bear_power_val < 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND Bull Power > 0 (weakening bullish momentum) AND volume filter
            elif wr_cross_below_20 and bull_power_val > 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -20 OR Bear Power becomes positive (bullish momentum returning)
                if wr > -20 or bear_power_val > 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -80 OR Bull Power becomes negative (bearish momentum returning)
                if wr < -80 or bull_power_val < 0:
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