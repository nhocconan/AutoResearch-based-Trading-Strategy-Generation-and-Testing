#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 1-day Elder Ray power confirmation and volume filter.
# The Alligator (Jaw/Teeth/Lips) identifies trend direction and strength via smoothed SMAs.
# Elder Ray (Bull/Bear Power) from 1-day confirms institutional buying/selling pressure.
# Volume > 1.5x average confirms participation, reducing false signals.
# Only trade when Alligator is aligned (no intertwining) and Elder Ray confirms direction.
# This combination aims for 20-30 trades per year per symbol (80-120 total over 4 years),
# staying within optimal range to minimize fee drift while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_len = 13
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = (df_1d['high'] - ema_1d).values
    bear_power = (ema_1d - df_1d['low']).values
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Williams Alligator on 4h: Jaw=SMMA(13,8), Teeth=SMMA(8,5), Lips=SMMA(5,3)
    def smma(source, period):
        # Smoothed Moving Average: first value = SMA, then SMMA = (prev*(period-1) + current)/period
        result = np.full_like(source, np.nan, dtype=np.float64)
        sma_period = np.convolve(source, np.ones(period)/period, mode='valid')
        if len(sma_period) == 0:
            return result
        result[period-1:len(sma_period)+period-1] = sma_period
        for i in range(len(sma_period)+period, len(source)):
            result[i] = (result[i-1]*(period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)  # SMMA(13,8) -> period 13
    teeth = smma(close, 8) # SMMA(8,5) -> period 8
    lips = smma(close, 5)  # SMMA(5,3) -> period 5
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 13)  # enough for Alligator and volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: check if not intertwined (trending market)
        # Jaw > Teeth > Lips = uptrend, Lips > Teeth > Jaw = downtrend
        jaw_above_teeth = jaw[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        
        uptrend_aligned = jaw_above_teeth and teeth_above_lips
        downtrend_aligned = lips_above_teeth and teeth_above_jaw
        
        # Elder Ray confirmation from 1-day
        bull_power_confirm = bull_power_aligned[i] > 0
        bear_power_confirm = bear_power_aligned[i] > 0
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Alligator uptrend aligned + bull power positive + volume
            if uptrend_aligned and bull_power_confirm and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Alligator downtrend aligned + bear power positive + volume
            elif downtrend_aligned and bear_power_confirm and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator loses alignment or bear power takes over
            if not uptrend_aligned or bear_power_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator loses alignment or bull power takes over
            if not downtrend_aligned or bull_power_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Alligator_ElderRay_Volume_v1"
timeframe = "4h"
leverage = 1.0