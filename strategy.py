#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d volume spike and 1d ADX trend filter.
# Long when price breaks above upper BB(20,2) AND volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below lower BB(20,2) AND volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price returns to middle BB (20-period SMA) or opposite band touch.
# Uses discrete position size 0.25. BB provides dynamic support/resistance, volume confirmation reduces false signals,
# and 1d ADX ensures we only trade in trending regimes. Target: 80-180 total trades over 4 years (20-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    if n < 20:
        return np.zeros(n)
    
    # Calculate SMA and standard deviation for BB
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    middle_band = sma_20  # 20-period SMA
    
    # === 1d Indicators: Volume MA and ADX ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period volume average on 1d
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) on 1d
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
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma_20_aligned[i]
        adx_val = adx_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1d ADX > 25 (trending regime)
        trend_filter = adx_val > 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or touches lower band
            if price <= middle or price <= lower:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or touches upper band
            if price >= middle or price >= upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper BB with volume and trend confirmation
            if price > upper and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below lower BB with volume and trend confirmation
            elif price < lower and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_BB20_2_1dVolumeSpike_1dADXTrend_V1"
timeframe = "6h"
leverage = 1.0