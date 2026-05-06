#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian channel breakout with 1d volume confirmation and 1w ADX trend filter
# - Uses 1w Donchian channel (20) for structural breakout levels
# - Uses 1d volume spike (2x 20-period MA) for confirmation
# - Uses 1w ADX > 25 to confirm trend strength
# - Enters long when price breaks above 1w Donchian high with volume spike and strong trend
# - Enters short when price breaks below 1w Donchian low with volume spike and strong trend
# - Exits when price crosses back below/above 1w Donchian middle or trend weakens (ADX < 20)
# - Designed to capture major trend moves with multi-timeframe confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1wDonchian_1dVolume_1wADX_Trend"
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
    
    # Get 1w data for Donchian channel and ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1w Donchian Channel (20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high and low
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 1w ADX (14)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1)) if 'close_1w' in locals() else np.abs(high_1w - np.roll(df_1w['close'].values, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1)) if 'close_1w' in locals() else np.abs(low_1w - np.roll(df_1w['close'].values, 1))
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
    
    close_1w = df_1w['close'].values
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
    donch_high_12h = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_12h = align_htf_to_ltf(prices, df_1w, donch_mid)
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filters (1d data aligned to 12h)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_20)
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or np.isnan(donch_mid_12h[i]) or
            np.isnan(adx_12h[i]) or np.isnan(volume_spike_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for strong trend (ADX > 25)
            strong_trend = adx_12h[i] > 25
            
            if strong_trend:
                # Long: price breaks above 1w Donchian high with volume spike
                if close[i] > donch_high_12h[i] and volume_spike_12h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1w Donchian low with volume spike
                elif close[i] < donch_low_12h[i] and volume_spike_12h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below 1w Donchian mid OR trend weakens (ADX < 20)
            if close[i] < donch_mid_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w Donchian mid OR trend weakens (ADX < 20)
            if close[i] > donch_mid_12h[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals