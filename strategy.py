#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian breakout with 1d volume confirmation and 1d ADX trend filter.
# Long when price breaks above 12h Donchian(20) upper band, 1d volume > 1.5x 20-period average, and 1d ADX > 25.
# Short when price breaks below 12h Donchian(20) lower band, 1d volume > 1.5x 20-period average, and 1d ADX > 25.
# Exit when price returns to 12h Donchian midline or ADX drops below 20 (trend weakening).
# Uses discrete position size 0.25. Donchian provides structure, volume confirms breakout validity,
# ADX filters weak trends. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper band: highest high over 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align 12h Donchian to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle_20)
    
    # Get 1d data once before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume MA (20) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1d Indicators: ADX (14) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # first value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] + (data[i] - result[i-1]) / period
        return result
    
    atr_14 = WilderSmooth(tr, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmooth(dx, 14)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midline or ADX drops below 20 (weakening trend)
            if price <= middle or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midline or ADX drops below 20 (weakening trend)
            if price >= middle or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average (1d)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Trend filter: ADX > 25 (strong trend)
            trend_filter = adx_val > 25
            
            # LONG: price breaks above upper band, volume spike, strong trend
            if price > upper and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below lower band, volume spike, strong trend
            elif price < lower and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_12hDonchian20_1dVolume_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0