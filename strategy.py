#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 1d ADX trend filter
# - Uses 12h Donchian channels (20-period) to identify structural breakouts
# - Uses 1d ADX (14) to confirm trend strength (ADX > 25) for breakout direction
# - Enters long when price breaks above 12h upper Donchian with volume spike in strong trend
# - Enters short when price breaks below 12h lower Donchian with volume spike in strong trend
# - Exits when price crosses back below/above 12h middle Donchian or trend weakens (ADX < 20)
# - Designed to capture major trend moves with proper filtering to avoid whipsaws
# - Target: 80-160 total trades over 4 years (20-40/year) with 0.25 position sizing

name = "4h_Donchian12h_ADX1d_Volume"
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
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 12h Donchian Channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Upper band (20-period high)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band (20-period low)
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band (average of upper and lower)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d ADX (14)
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
    
    # Smoothed values (Wilder's smoothing)
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
    
    # Align 12h indicators to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_4h = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Align 1d ADX to 4h timeframe
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(donchian_mid_4h[i]) or np.isnan(adx_4h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25)
            strong_trend = adx_4h[i] > 25
            
            if strong_trend:
                # Long: price breaks above 12h upper Donchian with volume spike
                if close[i] > donchian_high_4h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 12h lower Donchian with volume spike
                elif close[i] < donchian_low_4h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 12h middle Donchian OR trend weakens (ADX < 20)
            if close[i] < donchian_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h middle Donchian OR trend weakens (ADX < 20)
            if close[i] > donchian_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals