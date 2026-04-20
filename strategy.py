#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# - Long when price breaks above Donchian upper band (20-bar high) AND 1d ADX > 25 (trending market) AND volume > 1.5x 20-period average volume
# - Short when price breaks below Donchian lower band (20-bar low) AND 1d ADX > 25 AND volume > 1.5x 20-period average volume
# - Exit when price crosses the Donchian midpoint (10-bar average of high/low) OR ADX drops below 20 (trend weakening)
# - Designed for 4h timeframe with selective entries to avoid overtrading and capture strong trends
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Smooth TR, DM+ and DM- with Welles Wilder's smoothing (alpha=1/14)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    dm_plus_di = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    dm_minus_di = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    dx = np.where((dm_plus_di + dm_minus_di) != 0, 
                  100 * np.abs(dm_plus_di - dm_minus_di) / (dm_plus_di + dm_minus_di), 0)
    adx_1d = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_4h / avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        adx = adx_1d_aligned[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper AND ADX > 25 (strong trend) AND volume > 1.5x average
            if price > donchian_upper[i] and adx > 25 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower AND ADX > 25 AND volume > 1.5x average
            elif price < donchian_lower[i] and adx > 25 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian middle OR ADX drops below 20 (trend weakening)
            if price < donchian_middle[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian middle OR ADX drops below 20
            if price > donchian_middle[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_ADX_Volume_Filter"
timeframe = "4h"
leverage = 1.0