# NOTE: This strategy failed to compile due to a syntax error in the original code.
# The error was: invalid syntax in line '    for i in range(start_idx, n):'
# This was likely due to incorrect indentation or mixing of tabs and spaces.
# Please review the code structure carefully when reimplementing.
# The strategy logic remains valid but requires proper syntax correction.
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d volume spike and 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions. Readings below -80 indicate oversold,
# above -20 indicate overbought. We look for reversals from these extremes.
# Volume spike confirms institutional participation in the reversal.
# 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranges.
# This combination works in both bull and bear markets by trading reversals within strong trends.
# Targets 15-25 trades per year (~60-100 total over 4 years) to minimize fee drag.

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
    
    # Get 1d data for Williams %R, volume, and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on daily
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r.values)
    
    # Williams %R signals: oversold (< -80) or overbought (> -20)
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # Volume spike detection on 1d
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean()
    vol_spike = df_1d['volume'].values > (vol_ma.values * 2.0)
    
    # ADX trend filter on 1d
    # Calculate ADX(14) on daily
    plus_dm = np.zeros_like(df_1d['high'])
    minus_dm = np.zeros_like(df_1d['high'])
    tr = np.zeros_like(df_1d['high'])
    
    for i in range(1, len(df_1d)):
        plus_dm[i] = max(df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1], 0)
        minus_dm[i] = max(df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        tr[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
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
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    
    # Align all 1d indicators to 12h timeframe (use previous day's values)
    williams_oversold_12h = align_htf_to_ltf(prices, df_1d, williams_oversold)
    williams_overbought_12h = align_htf_to_ltf(prices, df_1d, williams_overbought)
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike)
    adx_strong_12h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_12h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient data for volume MA and Williams %R
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_oversold_12h[i]) or np.isnan(williams_overbought_12h[i]) or 
            np.isnan(vol_spike_12h[i]) or np.isnan(adx_strong_12h[i]) or np.isnan(adx_weak_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from oversold, volume spike, strong trend
            if williams_oversold_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from overbought, volume spike, strong trend
            elif williams_overbought_12h[i] and vol_spike_12h[i] and adx_strong_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reaches overbought or trend weakens
            if williams_overbought_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reaches oversold or trend weakens
            if williams_oversold_12h[i] or adx_weak_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals