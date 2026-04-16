#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w ADX trend filter.
# Long when price breaks above Donchian upper band (20-period) AND volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when price breaks below Donchian lower band (20-period) AND volume > 2.0x 20-period average AND 1w ADX > 25.
# Exit when price returns to Donchian middle band (20-period midpoint) or ATR(10) < ATR(30) (contracting volatility).
# Uses discrete position size 0.25. Donchian from 1d provides clear structure, volume confirmation reduces false breakouts,
# 1w ADX ensures we only trade in trending regimes. Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Donchian channels (20-period) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 1d timeframe (no alignment needed as primary TF is 1d)
    upper_aligned = upper_20
    lower_aligned = lower_20
    middle_aligned = middle_20
    
    # Get 1w data once before loop for volume and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1w Indicators: Volume MA and ADX(14) ===
    # Volume moving average (20-period)
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # True Range for ADX
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w indicators to 1d timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # ATR for volatility regime (using 1d data)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr_10 = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_10[i]) or np.isnan(atr_30[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        middle_val = middle_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        atr_10_val = atr_10[i]
        atr_30_val = atr_30[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or ATR contracts
            if price <= middle_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or ATR contracts
            if price >= middle_val or atr_10_val < atr_30_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2.0x 20-period average (stricter to reduce trades)
            vol_filter = vol > 2.0 * vol_ma_val
            
            # Trend filter: 1w ADX > 25 (trending regime)
            trend_filter = adx_val > 25
            
            # LONG: price breaks above Donchian upper band with volume and trend confirmation
            if price > upper_val and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with volume and trend confirmation
            elif price < lower_val and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wVolumeSpike_1wADXTrend_V1"
timeframe = "1d"
leverage = 1.0