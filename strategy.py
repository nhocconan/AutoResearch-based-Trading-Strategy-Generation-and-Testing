#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + ADX trend filter
# - Breakout above 20-period high or below 20-period low on 12h timeframe
# - Volume spike: current volume > 2.0 * average volume of last 20 periods on 1d
# - ADX > 25 to confirm trending market (avoid ranging markets)
# - Long on upward breakout with volume spike and ADX > 25
# - Short on downward breakout with volume spike and ADX > 25
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ADX(14) on 1d timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    dm_plus_safe = np.where(atr == 0, 1e-10, dm_plus_smooth)
    dm_minus_safe = np.where(atr == 0, 1e-10, dm_minus_smooth)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    di_plus = 100 * dm_plus_safe / atr_safe
    di_minus = 100 * dm_minus_safe / atr_safe
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilder_smooth(dx, 14)
    
    # Average volume for spike detection
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_condition = (volume_1d > 2.0 * avg_volume)
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike_condition.astype(float))
    
    # Calculate Donchian channels on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_12h[i]) or np.isnan(volume_spike_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        adx_val = adx_12h[i]
        vol_spike = volume_spike_12h[i] > 0.5  # Boolean condition
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + ADX > 25
            if price > donchian_high[i] and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + ADX > 25
            elif price < donchian_low[i] and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or ADX < 20 (trend weakening)
            if price < donchian_low[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or ADX < 20 (trend weakening)
            if price > donchian_high[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0