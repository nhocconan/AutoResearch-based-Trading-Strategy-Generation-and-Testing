#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) + 1d Volume Spike + 1d ADX Trend Filter
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect institutional buying/selling pressure.
# Combines with 1d volume surge to confirm institutional participation and 1d ADX > 25 to ensure trending conditions.
# Works in bull markets via bull power expansion and in bear markets via bear power expansion.
# Targets 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

name = "6h_ElderRay_1dVolume_1dADX"
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
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) on daily closes
    close_1d_series = pd.Series(close_1d)
    ema_13 = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike detection on 1d (need ~2 days for MA)
    vol_ma_1d = pd.Series(volume).rolling(window=48, min_periods=48).mean()  # 48 * 6h = 12d approx
    vol_spike = volume > (vol_ma_1d.values * 2.0)
    
    # ADX trend filter on 1d
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
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * (plus_dm14 / tr14), 0)
    minus_di14 = np.where(tr14 != 0, 100 * (minus_dm14 / tr14), 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilder_smooth(dx, 14)
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    adx_strong_6h = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_6h = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 48  # Ensure sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(vol_spike[i]) or 
            np.isnan(adx_strong_6h[i]) or np.isnan(adx_weak_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bull power expansion, volume spike, strong trend
            if bull_power_6h[i] > 0 and vol_spike[i] and adx_strong_6h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power expansion, volume spike, strong trend
            elif bear_power_6h[i] < 0 and vol_spike[i] and adx_strong_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power contraction or trend weakens
            if bull_power_6h[i] <= 0 or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power contraction or trend weakens
            if bear_power_6h[i] >= 0 or adx_weak_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals