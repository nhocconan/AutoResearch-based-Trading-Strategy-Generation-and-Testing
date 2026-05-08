#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume spike
# Williams %R identifies overbought/oversold conditions. Readings below -80 indicate oversold,
# above -20 indicate overbought. We buy when %R crosses above -80 from below in an uptrend
# (ADX > 25) with volume confirmation. Sell when %R crosses below -20 from above.
# This mean-reversion strategy works in ranging markets, while the ADX filter ensures we
# only take trades in the direction of the daily trend to avoid counter-trend whipsaws.
# Targets 20-30 trades per year (~80-120 total over 4 years) with controlled risk.

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
    
    # Get 1d data for ADX and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 6h data (14-period)
    def williams_r(high_arr, low_arr, close_arr, period=14):
        highest_high = np.full_like(high_arr, np.nan)
        lowest_low = np.full_like(low_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            highest_high[i] = np.max(high_arr[i-period+1:i+1])
            lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low),
                      -50)  # Neutral when no range
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Calculate ADX(14) on daily data
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
    
    # ADX thresholds
    adx_strong = adx > 25  # Strong trend
    
    # Volume spike detection on 1d (24-period MA for ~4 days)
    vol_ma = np.full_like(volume, np.nan)
    for i in range(24, len(volume)):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Align 1d indicators to 6h timeframe
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    vol_spike_6h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 14)  # Ensure sufficient data for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr[i]) or np.isnan(adx_strong_6h[i]) or np.isnan(vol_spike_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below, volume spike, strong trend
            if i > 0 and wr[i-1] <= -80 and wr[i] > -80 and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above, volume spike, strong trend
            elif i > 0 and wr[i-1] >= -20 and wr[i] < -20 and vol_spike_6h[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -20 from above or trend weakens
            if i > 0 and wr[i-1] >= -20 and wr[i] < -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -80 from below or trend weakens
            if i > 0 and wr[i-1] <= -80 and wr[i] > -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals