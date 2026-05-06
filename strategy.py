#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h VWAP deviation with 1d ADX trend filter and volume confirmation
# - Long when price > 12h VWAP + 1.5*ATR, 1d ADX > 25, and volume > 1.5x 20-period average
# - Short when price < 12h VWAP - 1.5*ATR, 1d ADX > 25, and volume > 1.5x 20-period average
# - Exit when price crosses back below/above 12h VWAP
# - Designed to capture strong trending moves with institutional volume confirmation
# - Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_VWAP_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h VWAP (Volume Weighted Average Price)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h = vwap_12h.values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX (Average Directional Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (14-period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan)
    for i in range(13, len(dx)):
        if i == 13:
            adx[i] = np.nanmean(dx[1:14])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 4h ATR for entry threshold
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h VWAP and 1d ADX to 4h timeframe
    vwap_4h = align_htf_to_ltf(prices, df_12h, vwap_12h)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(vwap_4h[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > VWAP + 1.5*ATR, ADX > 25, volume spike
            if close[i] > vwap_4h[i] + 1.5 * atr_4h[i] and adx_4h[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < VWAP - 1.5*ATR, ADX > 25, volume spike
            elif close[i] < vwap_4h[i] - 1.5 * atr_4h[i] and adx_4h[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below VWAP
            if close[i] < vwap_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above VWAP
            if close[i] > vwap_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals