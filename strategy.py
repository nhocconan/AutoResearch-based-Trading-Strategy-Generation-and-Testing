#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bull/Bear Power (Elder Ray) + 12h Supertrend + volume confirmation
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# Go long when Bull Power > 0 and Bear Power < 0 with rising 12h Supertrend and volume spike
# Go short when Bear Power > 0 and Bull Power < 0 with falling 12h Supertrend and volume spike
# Exit when power signals reverse or volume drops below average
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by using Supertrend for trend direction and Elder Ray for momentum

name = "6h_ElderRay_12hSupertrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power and Bear Power (13-period EMA)
    ema_len = 13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 1.5)
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR using Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period+1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    if not np.isnan(atr[atr_period]):
        supertrend[atr_period] = lower_band[atr_period]
        direction[atr_period] = 1 if close_12h[atr_period] > supertrend[atr_period] else -1
    
    for i in range(atr_period + 1, len(close_12h)):
        if np.isnan(atr[i]) or np.isnan(supertrend[i-1]):
            continue
            
        if close_12h[i] > upper_band[i]:
            direction[i] = 1
        elif close_12h[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and supertrend[i-1] < lower_band[i]:
                supertrend[i] = lower_band[i]
            elif direction[i] == -1 and supertrend[i-1] > upper_band[i]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(supertrend[i], lower_band[i])
        else:
            supertrend[i] = min(supertrend[i], upper_band[i])
    
    # Align Supertrend direction to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_len, 20, atr_period + 10)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_spike[i]) or np.isnan(supertrend_dir_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, uptrend, volume spike
            if bull_power[i] > 0 and bear_power[i] < 0 and supertrend_dir_aligned[i] == 1 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0, Bull Power < 0, downtrend, volume spike
            elif bear_power[i] > 0 and bull_power[i] < 0 and supertrend_dir_aligned[i] == -1 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 or trend changes or volume drops
            if bull_power[i] <= 0 or bear_power[i] >= 0 or supertrend_dir_aligned[i] != 1 or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 or Bull Power >= 0 or trend changes or volume drops
            if bear_power[i] <= 0 or bull_power[i] >= 0 or supertrend_dir_aligned[i] != -1 or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals