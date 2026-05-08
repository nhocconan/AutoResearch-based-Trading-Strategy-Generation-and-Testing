#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume spike
# Williams %R identifies overbought/oversold conditions. Readings below -80 indicate oversold
# (long opportunity), above -20 indicate overbought (short opportunity). We require
# 1d ADX > 25 to ensure we only trade in strong trends, avoiding whipsaws in ranges.
# Volume spike confirms institutional participation. This combination works in both
# bull and bear markets by trading mean reversions within strong trends.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

name = "6h_WilliamsR_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R(14) calculation
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-13:i+1])
        lowest_low[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close_1d) / (highest_high - lowest_low),
                          -50)
    
    # ADX(14) calculation on daily
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0)
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Wilder smoothing
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    # Williams %R signals: oversold < -80, overbought > -20
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # ADX trend filter
    adx_strong = adx > 25
    adx_weak = adx < 20
    
    # Volume spike detection on 1d
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean()  # 24 * 6h = 4d approx
    vol_spike = volume > (vol_ma.values * 2.0)
    
    # Align all indicators to 6h timeframe
    williams_oversold_6h = align_htf_to_ltf(prices, df_1d, williams_oversold)
    williams_overbought_6h = align_htf_to_ltf(prices, df_1d, williams_overbought)
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_6h = align_htf_to_ltf(prices, df_1d, adx_weak)
    vol_spike_6h = vol_spike  # already 6h data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_oversold_6h[i]) or np.isnan(williams_overbought_6h[i]) or 
            np.isnan(adx_strong_6h[i]) or np.isnan(adx_weak_6h[i]) or 
            np.isnan(vol_spike_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold, volume spike, strong trend
            if williams_oversold_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought, volume spike, strong trend
            elif williams_overbought_6h[i] and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to overbought or trend weakens
            if williams_overbought_6h[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to oversold or trend weakens
            if williams_oversold_6h[i] or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals