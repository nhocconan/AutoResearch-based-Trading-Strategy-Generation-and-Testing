#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily ADX filter and volume confirmation.
# Long when: Price breaks above upper Donchian channel, daily ADX > 25 (trending), volume > 1.3x 20-period average
# Short when: Price breaks below lower Donchian channel, daily ADX > 25 (trending), volume > 1.3x 20-period average
# Exit when: Price crosses back through the middle of Donchian channel (mean of upper and lower)
# Donchian channels provide clear breakout levels, ADX filters for trending markets only, volume confirms breakout strength.
# Target: 15-25 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "12h_Donchian20_ADX_Volume"
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
    
    # 1-day data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    
    upper_20 = high_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_series.rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate ADX(14) on daily data for trend strength filter
    # ADX calculation requires +DI, -DI, and DX
    # +DM = max(0, high_t - high_{t-1}) if high_t - high_{t-1} > low_{t-1} - low_t else 0
    # -DM = max(0, low_{t-1} - low_t) if low_{t-1} - low_t > high_t - high_{t-1} else 0
    # TR = max(high-low, abs(high-close_{t-1}), abs(low-close_{t-1}))
    # +DM_smooth = smoothed +DM (Wilder's smoothing)
    # -DM_smooth = smoothed -DM (Wilder's smoothing)
    # TR_smooth = smoothed TR (Wilder's smoothing)
    # +DI = 100 * (+DM_smooth / TR_smooth)
    # -DI = 100 * (-DM_smooth / TR_smooth)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX
    
    # Calculate directional movement
    high_diff = high_1d[1:] - high_1d[:-1]
    low_diff = low_1d[:-1] - low_1d[1:]  # low_{t-1} - low_t
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period_adx = 14
    if len(plus_dm) >= period_adx:
        plus_dm_smooth = wilders_smoothing(plus_dm, period_adx)
        minus_dm_smooth = wilders_smoothing(minus_dm, period_adx)
        tr_smooth = wilders_smoothing(tr, period_adx)
        
        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0.0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0.0)
        
        # DX calculation
        di_sum = plus_di + minus_di
        dx = np.where(di_sum != 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0.0)
        
        # ADX is smoothed DX
        adx = wilders_smoothing(dx, period_adx)
        
        # Prepend NaN for the first element (since we lost one in diff)
        plus_di = np.concatenate([np.array([np.nan]), plus_di])
        minus_di = np.concatenate([np.array([np.nan]), minus_di])
        adx = np.concatenate([np.array([np.nan] * period_adx), adx])
    else:
        adx = np.full_like(high_1d, np.nan)
        plus_di = np.full_like(high_1d, np.nan)
        minus_di = np.full_like(high_1d, np.nan)
    
    # Align 1D data to 12H timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20 + 14 + 14  # Wait for Donchian (20) + ADX (14+14 smoothing) 
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above upper Donchian, ADX > 25 (trending), volume spike
            if (price > upper and close[i-1] <= upper and 
                adx_val > 25 and vol > 1.3 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower Donchian, ADX > 25 (trending), volume spike
            elif (price < lower and close[i-1] >= lower and 
                  adx_val > 25 and vol > 1.3 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below middle of Donchian channel
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above middle of Donchian channel
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals