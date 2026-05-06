#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with 1w volume confirmation and 1w ADX trend filter
# - Uses 1w Donchian channel (20-period) for breakout signals
# - Uses 1w ADX > 25 to confirm trend strength in the breakout direction
# - Uses 1w volume > 1.5x 20-period average for volume confirmation
# - Enters long when price breaks above 1w upper Donchian with trend and volume confirmation
# - Enters short when price breaks below 1w lower Donchian with trend and volume confirmation
# - Exits when price crosses back below/above 1w middle (median) of the Donchian channel
# - Designed to capture strong trend continuations after weekly consolidation
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wDonchian_ADX_Volume_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian, ADX, and volume calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w Donchian Channel (20-period)
    donchian_window = 20
    upper_donchian = pd.Series(high_1w).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_donchian = pd.Series(low_1w).rolling(window=donchian_window, min_periods=donchian_window).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2.0  # Median of the channel
    
    # Calculate 1w ADX (14-period)
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
    
    # Calculate 1w volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_factor = 1.5  # Volume must be 1.5x average
    
    # Align 1w indicators to 1d timeframe
    upper_donchian_1d = align_htf_to_ltf(prices, df_1w, upper_donchian)
    lower_donchian_1d = align_htf_to_ltf(prices, df_1w, lower_donchian)
    middle_donchian_1d = align_htf_to_ltf(prices, df_1w, middle_donchian)
    adx_1d = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_1d = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(upper_donchian_1d[i]) or np.isnan(lower_donchian_1d[i]) or 
            np.isnan(middle_donchian_1d[i]) or np.isnan(adx_1d[i]) or 
            np.isnan(vol_ma_20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25) and volume confirmation
            strong_trend = adx_1d[i] > 25
            volume_confirmed = volume[i] > (volume_factor * vol_ma_20_1d[i])
            
            if strong_trend and volume_confirmed:
                # Long: price breaks above 1w upper Donchian
                if close[i] > upper_donchian_1d[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1w lower Donchian
                elif close[i] < lower_donchian_1d[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1w middle Donchian
            if close[i] < middle_donchian_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w middle Donchian
            if close[i] > middle_donchian_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals