#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend + volume confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trending vs ranging markets
# In trending markets (JAW > TEETH > LIPS for uptrend, JAW < TEETH < LIPS for downtrend),
# we enter breakouts in the direction of the 1d EMA34 trend with volume confirmation
# This avoids whipsaws in ranging markets and captures strong trends in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Williams Alligator: SMoothed Moving Average (SMMA)
    # JAW: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    # SMMA is similar to EMA but with different smoothing
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(high + low / 2, 13)  # Using median price for Alligator
    teeth = smma(high + low / 2, 8)
    lips = smma(high + low / 2, 5)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator trends:
        # Uptrend: LIPS > TEETH > JAW
        # Downtrend: JAW > TEETH > LIPS
        # Ranging: lines are intertwined
        is_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        is_downtrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:
            # Long conditions: Alligator uptrend AND price > 1d EMA34 AND volume spike
            if is_uptrend and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator downtrend AND price < 1d EMA34 AND volume spike
            elif is_downtrend and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator trend changes to downtrend OR price crosses below 1d EMA34
            if is_downtrend or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator trend changes to uptrend OR price crosses above 1d EMA34
            if is_uptrend or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals