#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 1d ADX trend filter
# - Uses 12h Donchian channels (20-period) to identify structural breakouts
# - Requires volume > 1.5x 20-period average for confirmation
# - Uses 1d ADX > 25 to ensure strong trend context
# - Exits when price crosses the Donchian midpoint or trend weakens (ADX < 20)
# - Designed to capture strong trending moves with institutional volume
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hDonchian_1dADX_Volume"
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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper and lower bands
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 1d ADX (14)
    # True Range
    tr1 = high_12h - low_12h  # Use 12h data for TR consistency
    tr2 = np.abs(high_12h - np.roll(low_12h, 1))
    tr3 = np.abs(low_12h - np.roll(high_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
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
    
    # Align 12h indicators to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_4h = align_htf_to_ltf(prices, df_12h, donch_mid)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or np.isnan(donch_mid_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25) and volume confirmation
            strong_trend = adx_4h[i] > 25
            
            if strong_trend and volume_confirm[i]:
                # Long: price breaks above 12h Donchian high
                if close[i] > donch_high_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 12h Donchian low
                elif close[i] < donch_low_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR trend weakens (ADX < 20)
            if close[i] < donch_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR trend weakens (ADX < 20)
            if close[i] > donch_mid_4h[i] or adx_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals