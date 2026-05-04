#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR(14) volatility filter and volume confirmation (>1.5x 20 EMA volume)
# Uses Donchian channels from prior completed 4h bar for structure (breakout = new 20-period high/low)
# 1d ATR(14) filter ensures we only trade when volatility is elevated (>1.0x ATR median) to avoid choppy markets
# Volume confirmation ensures breakout has sufficient participation (>1.5x average volume)
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 4h timeframe
# Works in both bull (breakouts continuation) and bear (breakdowns continuation) markets
# Focus on BTC/ETH by requiring volatility and volume filters (avoids false breakouts in low-vol regimes)

name = "4h_Donchian20_VolATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR(14) volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough data for ATR calculation
        return np.zeros(n)
    
    # Calculate 1d ATR(14) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if np.isnan(atr_14_1d[i-1]):
            atr_14_1d[i] = np.nanmean(tr[i-13:i+1])
        else:
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    atr_14_1d_shifted = np.roll(atr_14_1d, 1)
    atr_14_1d_shifted[0] = np.nan
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d_shifted)
    
    # Calculate ATR median for volatility regime filter (using prior completed 1d bars)
    atr_median = np.full_like(atr_14_1d_aligned, np.nan)
    for i in range(50, len(atr_14_1d_aligned)):  # Need 50 bars for median calculation
        window = atr_14_1d_aligned[i-49:i+1]
        valid_vals = window[~np.isnan(window)]
        if len(valid_vals) >= 10:
            atr_median[i] = np.median(valid_vals)
    
    # Get 4h data for Donchian channels (prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need enough data for Donchian calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) channels: upper = max(high, 20), lower = min(low, 20)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high_4h, 20)
    donchian_lower = rolling_min(low_4h, 20)
    
    # Shift by 1 to use only prior completed 4h bar (no look-ahead)
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_shifted)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when current ATR > 1.0 * ATR median (elevated volatility)
        vol_filter = atr_14_1d_aligned[i] > (1.0 * atr_median[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + volume spike + volatility filter
            if close[i] > donchian_upper_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]) and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + volume spike + volatility filter
            elif close[i] < donchian_lower_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]) and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals