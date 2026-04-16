#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 4h volume confirmation and 1d ADX trend filter.
# Long when price breaks above Camarilla R3 AND 4h volume > 2.0x 20-period average AND 1d ADX > 20.
# Short when price breaks below Camarilla S3 AND 4h volume > 2.0x 20-period average AND 1d ADX > 20.
# Exit when price returns to Camarilla Pivot (PP).
# Uses discrete position size 0.30. Camarilla levels from 1d provide strong support/resistance.
# Volume confirmation reduces false breakouts, 1d ADX ensures trending regime.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla levels (based on previous day) ===
    # Calculate Camarilla for each 1d bar using previous day's OHLC
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(high_1d), np.nan)
    camarilla_pp = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Camarilla formulas
        range_val = phigh - plow
        camarilla_pp[i] = (phigh + plow + pclose) / 3.0
        camarilla_r3[i] = camarilla_pp[i] + range_val * 1.1 / 4.0
        camarilla_s3[i] = camarilla_pp[i] - range_val * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Get 4h data for volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for ADX filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d_adx = df_1d['high'].values
    low_1d_adx = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = high_1d_adx - low_1d_adx
    tr2 = np.abs(high_1d_adx - np.roll(close_1d_adx, 1))
    tr3 = np.abs(low_1d_adx - np.roll(close_1d_adx, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d_adx - np.roll(high_1d_adx, 1)) > (np.roll(low_1d_adx, 1) - low_1d_adx),
                       np.maximum(high_1d_adx - np.roll(high_1d_adx, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d_adx, 1) - low_1d_adx) > (high_1d_adx - np.roll(high_1d_adx, 1)),
                        np.maximum(np.roll(low_1d_adx, 1) - low_1d_adx, 0), 0)
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
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma_val = vol_ma_20[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using same timeframe volume)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1d ADX > 20 (trending regime)
        trend_filter = adx_val > 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Camarilla Pivot Point
            if price <= pp_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Camarilla Pivot Point
            if price >= pp_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Camarilla R3 with volume and trend confirmation
            if price > r3_val and vol_filter and trend_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S3 with volume and trend confirmation
            elif price < s3_val and vol_filter and trend_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_CamarillaR3S3_4hVolumeSpike_1dADXTrend_V1"
timeframe = "4h"
leverage = 1.0