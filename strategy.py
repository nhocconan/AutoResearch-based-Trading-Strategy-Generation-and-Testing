#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with daily Williams Alligator (SMAs) + Elder Ray (Bull/Bear Power) + volume confirmation
# Williams Alligator: Jaw (13 SMMA), Teeth (8 SMMA), Lips (5 SMMA) - alignment indicates trend
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# In trending markets (Alligator aligned), Elder Ray shows momentum strength
# Volume confirms conviction. Works in bull/bear by following trend direction.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Alligator_ElderRay_Volume"
timeframe = "6h"
leverage = 1.0

def _smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
    smma = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        if i < period - 1:
            continue
        if i == period - 1:
            smma[i] = sma[i]
        else:
            smma[i] = (smma[i-1] * (period-1) + series[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator: SMMA of median price (HL/2)
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = _smma(median_price.values, 13)  # Jaw: 13-period SMMA
    teeth = _smma(median_price.values, 8)   # Teeth: 8-period SMMA
    lips = _smma(median_price.values, 5)    # Lips: 5-period SMMA
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = ema13 - df_1d['low'].values
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume filter: current 6h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Volume MA and EMA13
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_filter = volume_filter[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        is_uptrend = lips_val > teeth_val > jaw_val
        is_downtrend = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Enter long: uptrend + bullish Elder Ray + volume
            if is_uptrend and bull_power_val > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + bearish Elder Ray + volume
            elif is_downtrend and bear_power_val > 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend change or weakening bull power
            if not is_uptrend or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend change or weakening bear power
            if not is_downtrend or bear_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals