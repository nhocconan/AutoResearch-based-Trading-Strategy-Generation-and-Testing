#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel (20) breakout with 4h volume spike (2.0x 20-bar average) and 1d ADX > 25 trend filter.
# Long when price breaks above Donchian upper band AND volume > 2.0x 20-bar avg AND 1d ADX > 25.
# Short when price breaks below Donchian lower band AND volume > 2.0x 20-bar avg AND 1d ADX > 25.
# Exit when price returns to Donchian middle band (mean of upper/lower) or ATR(10) < ATR(30) (volatility contraction).
# Uses discrete position size 0.25. Donchian provides structure, volume confirmation reduces false signals,
# and 1d ADX ensures we only trade in trending regimes. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_rolling_max
    lower_band = low_rolling_min
    middle_band = (upper_band + lower_band) / 2.0
    
    # === 4h Indicators: ATR(10) and ATR(30) for volatility filter ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # === 4h Indicators: Volume MA(20) for volume spike filter ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend filter ===
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    # Smoothed values
    tr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr_14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr_14_1d
    
    # ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    dx_1d = np.where(np.isnan(dx_1d), 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or 
            np.isnan(atr_10[i]) or np.isnan(atr_30[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        middle_val = middle_band[i]
        atr_10_val = atr_10[i]
        atr_30_val = atr_30[i]
        vol_ma_20_val = vol_ma_20[i]
        adx_val = adx_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average
        vol_filter = vol > 2.0 * vol_ma_20_val if vol_ma_20_val > 0 else False
        
        # Trend filter: 1d ADX > 25 (trending regime)
        trend_filter = adx_val > 25
        
        # Volatility filter: ATR(10) > ATR(30) (expanding volatility)
        vol_filter_atr = atr_10_val > atr_30_val
        
        # Combined filters: volume AND trend AND volatility expansion
        combined_filter = vol_filter and trend_filter and vol_filter_atr
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to middle band or volatility contracts
            if price <= middle_val or not vol_filter_atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to middle band or volatility contracts
            if price >= middle_val or not vol_filter_atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper band with combined filter confirmation
            if price > upper_val and combined_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with combined filter confirmation
            elif price < lower_val and combined_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_4hVolumeSpike_1dADXTrend_ATRVolFilter_V1"
timeframe = "4h"
leverage = 1.0