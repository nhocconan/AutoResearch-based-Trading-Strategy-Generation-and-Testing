#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Reversal with 1d Elder Ray Power confirmation and volume spike.
Long when Williams %R < -80 (oversold) AND 1d Bear Power < 0 (bearish momentum weakening) AND volume > 2.0x 20-period MA.
Short when Williams %R > -20 (overbought) AND 1d Bull Power > 0 (bullish momentum weakening) AND volume > 2.0x 20-period MA.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR EMA(13) crossover.
Uses 1d HTF for Elder Ray to filter counter-trend trades, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Williams %R identifies reversals, Elder Ray confirms momentum shift, volume validates strength.
Designed to work in both bull and bear markets by catching overextended moves reversing.
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
    
    # Calculate 4h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Calculate 1d Elder Ray Power (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 4h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA13 for exit signal
    ema_13_4h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 20)  # Williams %R, Elder Ray EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(ema_13_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        ema_13 = ema_13_4h[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (high threshold to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND Bear Power < 0 (weakening bearish) AND volume spike
            if wr < -80 and bear_power < 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND Bull Power > 0 (weakening bullish) AND volume spike
            elif wr > -20 and bull_power > 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 OR EMA13 turns down
                if wr > -50 or (i >= start_idx + 1 and ema_13 < ema_13_4h[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 OR EMA13 turns up
                if wr < -50 or (i >= start_idx + 1 and ema_13 > ema_13_4h[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dElderRay_Power_VolumeSpike"
timeframe = "4h"
leverage = 1.0