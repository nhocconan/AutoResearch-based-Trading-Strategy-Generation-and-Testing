#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 12h Elder Ray power filter and 1d volume spike confirmation.
Long when Williams %R(14) < -80 (oversold) AND 12h Bull Power > 0 AND 1d volume > 1.5x 20-period MA.
Short when Williams %R(14) > -20 (overbought) AND 12h Bear Power < 0 AND 1d volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 12h HTF for Elder Ray trend filter to avoid counter-trend trades, 1d volume for momentum confirmation.
Williams %R is effective in both trending and ranging markets, Elder Ray measures bull/bear power.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate Williams %R(14)
    williams_r = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 12h Elder Bull/Bear Power (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA13 for Elder Ray
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Calculate 1d volume MA (20-period) for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 13, 20)  # Williams %R, Elder Ray EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Volume filter: 1d volume > 1.5x 20-period MA
        vol_filter = volume_1d[i // (60//15)] > 1.5 * vol_ma_val if i // (60//15) < len(volume_1d) else False
        # Simplified: use current bar's volume relative to aligned MA
        vol_filter = prices['volume'].iloc[i] > 1.5 * vol_ma_val
        
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
            # Exit when Williams %R crosses -50 (centerline)
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50
                if wr > -50:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_12hElderRay_Power_VolumeFilter"
timeframe = "6h"
leverage = 1.0