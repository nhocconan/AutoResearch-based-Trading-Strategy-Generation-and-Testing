#!/usr/bin/env python3
# 1h_SuperTrend_4hTrendFilter_VolumeSpike
# Hypothesis: SuperTrend on 1h with 4h trend filter (EMA50) and volume spike confirmation.
# SuperTrend adapts to volatility, 4h EMA50 filters trend direction, volume spike confirms momentum.
# Designed for low trade frequency (15-35/year) to minimize fee drag. Works in bull/bear via trend filter.

name = "1h_SuperTrend_4hTrendFilter_VolumeSpike"
timeframe = "1h"
leverage = 1.0

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
    
    # SuperTrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])  # Skip first NaN
        for i in range(atr_period, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr[i-1]):
                atr[i] = (multiplier * tr[i] + (multiplier - 1) * atr[i-1]) / multiplier
            else:
                atr[i] = np.nan
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize SuperTrend
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Start calculation after ATR is valid
    start_st = atr_period
    if start_st < len(close):
        supertrend[start_st] = upper_band[start_st]
        direction[start_st] = 1
        
        for i in range(start_st + 1, len(close)):
            if not np.isnan(close[i]) and not np.isnan(supertrend[i-1]):
                if close[i] > supertrend[i-1]:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
                    direction[i] = -1
            else:
                supertrend[i] = np.nan
                direction[i] = np.nan
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA50 on 4h
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    
    # Align 4h EMA50 to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume / 24-period average (24*1h = 1 day)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(atr_period + 1, 24, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: SuperTrend uptrend AND 4h EMA50 uptrend (close > EMA) AND volume spike
            if (direction[i] == 1 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Enter short: SuperTrend downtrend AND 4h EMA50 downtrend (close < EMA) AND volume spike
            elif (direction[i] == -1 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.20
            else:
                # Exit long: SuperTrend downtrend OR 4h trend reversal (close < EMA)
                if direction[i] == -1 or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = -0.20
            else:
                # Exit short: SuperTrend uptrend OR 4h trend reversal (close > EMA)
                if direction[i] == 1 or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals