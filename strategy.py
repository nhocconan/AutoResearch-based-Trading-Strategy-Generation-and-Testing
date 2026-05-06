#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1w Donchian breakout with 1d volume confirmation and 1d ADX trend filter
# - Uses 1w Donchian channels (20-period) to identify structural breaks
# - Uses 1d ADX > 25 to confirm trend strength in direction of breakout
# - Uses 1d volume spike (>2x 20-period MA) for confirmation
# - Enters long when price breaks above 1w high with volume spike and strong trend
# - Enters short when price breaks below 1w low with volume spike and strong trend
# - Exits when price crosses back below/above 1w close or ADX falls below 20
# - Designed to capture major trend changes with multi-timeframe confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1wDonchian_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Get 1d data for ADX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper and lower bands
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_close = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14)
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
    
    # Align 1w indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_close_12h = align_htf_to_ltf(prices, df_1w, donchian_close)
    
    # Align 1d indicators to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or np.isnan(donchian_close_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25)
            strong_trend = adx_12h[i] > 25
            
            if strong_trend:
                # Long: price breaks above 1w high with volume spike
                if close[i] > donchian_high_12h[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1w low with volume spike
                elif close[i] < donchian_low_12h[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1w close OR trend weakens (ADX < 20)
            if close[i] < donchian_close_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w close OR trend weakens (ADX < 20)
            if close[i] > donchian_close_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals