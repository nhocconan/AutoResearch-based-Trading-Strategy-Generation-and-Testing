#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to identify strength of buyers/sellers
# 1d ADX > 25 filters for trending markets only (avoids whipsaws in ranging conditions)
# Volume confirmation (>1.5x 20 EMA volume) ensures institutional participation
# Works in bull markets (buy when bull power rising + ADX trending) and bear markets (sell when bear power falling + ADX trending)
# Discrete sizing 0.25 minimizes fee churn while providing meaningful exposure
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe

name = "6h_ElderRay_1dADX_VolumeConfirm"
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) - measures trend strength
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[period-1:2*period-1]) if 2*period-1 <= len(data) else np.nanmean(data[period-1:])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_14 = np.where(atr_14 != 0, 100 * dm_plus_14 / atr_14, 0)
    di_minus_14 = np.where(atr_14 != 0, 100 * dm_minus_14 / atr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus_14 + di_minus_14) != 0, 
                  100 * np.abs(di_plus_14 - di_minus_14) / (di_plus_14 + di_minus_14), 0)
    adx_14 = wilders_smoothing(dx, 14)
    
    # Shift ADX by 1 to use only prior completed 1d bar
    adx_14_shifted = np.roll(adx_14, 1)
    adx_14_shifted[0] = np.nan
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_shifted)
    
    # Get 6h data for Elder Ray calculation
    # Calculate EMA13 of close for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power rising (today > yesterday) AND ADX > 25 (trending) AND volume spike
            if (i > 0 and bull_power[i] > bull_power[i-1] and 
                adx_14_aligned[i] > 25 and volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power falling (today < yesterday) AND ADX > 25 (trending) AND volume spike
            elif (i > 0 and bear_power[i] < bear_power[i-1] and 
                  adx_14_aligned[i] > 25 and volume[i] > (1.5 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power falling OR ADX < 20 (trend weakening)
            if bull_power[i] < bull_power[i-1] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power rising OR ADX < 20 (trend weakening)
            if bear_power[i] > bear_power[i-1] or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals