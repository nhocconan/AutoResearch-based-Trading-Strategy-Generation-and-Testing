#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with 1w ADX trend filter and volume confirmation
# - Uses 1d Donchian(20) breakout for entry signals
# - Uses 1w ADX > 25 to confirm strong trend direction
# - Enters long when price breaks above 1d upper Donchian with volume spike in strong uptrend
# - Enters short when price breaks below 1d lower Donchian with volume spike in strong downtrend
# - Exits when price returns to 1d middle Donchian band or trend weakens (ADX < 20)
# - Designed to capture strong trend continuations after consolidation with multi-timeframe confirmation
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_1dDonchian_1wADX_Volume"
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
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_donchian = (upper_donchian + lower_donchian) / 2
    
    # Calculate 1w ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 4h timeframe
    upper_donchian_4h = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_4h = align_htf_to_ltf(prices, df_1d, lower_donchian)
    middle_donchian_4h = align_htf_to_ltf(prices, df_1d, middle_donchian)
    
    # Align 1w ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian_4h[i]) or np.isnan(lower_donchian_4h[i]) or np.isnan(middle_donchian_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25)
            strong_trend = adx_4h[i] > 25
            
            if strong_trend:
                # Long: price breaks above 1d upper Donchian with volume spike
                if close[i] > upper_donchian_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d lower Donchian with volume spike
                elif close[i] < lower_donchian_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to middle Donchian OR trend weakens (ADX < 20)
            if close[i] < middle_donchian_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle Donchian OR trend weakens (ADX < 20)
            if close[i] > middle_donchian_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals