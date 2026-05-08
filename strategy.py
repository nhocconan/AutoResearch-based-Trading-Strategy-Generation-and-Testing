#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion + 1d volume spike + 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions. In strong trends (ADX>25),
# extreme readings often precede pullbacks, offering mean-reversion entries.
# Volume spike confirms institutional interest in the move.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by only trading in the direction of the trend.

name = "12h_WilliamsR_1dVolume_1dADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14) on 12h
    lookback = 14
    williams_r = np.full_like(high, np.nan)
    
    for i in range(lookback, n):
        highest_high = np.max(high[i-lookback:i+1])
        lowest_low = np.min(low[i-lookback:i+1])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 2.0)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily
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
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(vol_spike[i]) or 
            np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80), volume spike, strong trend
            if williams_r[i] < -80 and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20), volume spike, strong trend
            elif williams_r[i] > -20 and vol_spike[i] and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend weakens
            if williams_r[i] > -50 or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend weakens
            if williams_r[i] < -50 or adx_weak_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals