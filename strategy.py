#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot (R3/S3) breakout with 4h volume spike and 1d ADX trend filter.
# Long when price breaks above Camarilla R3 AND volume > 1.8x 20-period 4h average AND 1d ADX > 20.
# Short when price breaks below Camarilla S3 with same filters.
# Exit when price returns to Camarilla pivot point (PP).
# Uses discrete position size 0.20. Uses 4h/1d for signal direction, 1h only for entry timing.
# Session filter: 08-20 UTC to avoid low-volume periods. Target: 80-160 total trades over 4 years (20-40/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data once before loop for Camarilla pivot and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 4h Indicators: Camarilla pivot levels (based on previous 4h bar) ===
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R3 = Close + (High - Low) * 1.1 / 4
    # S3 = Close - (High - Low) * 1.1 / 4
    pp = (high_4h + low_4h + close_4h) / 3.0
    r3 = close_4h + (high_4h - low_4h) * 1.1 / 4.0
    s3 = close_4h - (high_4h - low_4h) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (use previous bar's values)
    pp_aligned_4h = align_htf_to_ltf(prices, df_4h, pp)
    r3_aligned_4h = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned_4h = align_htf_to_ltf(prices, df_4h, s3)
    
    # 4h volume average (20-period)
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend filter ===
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
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1d timeframe
    adx_aligned_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned_4h[i]) or np.isnan(r3_aligned_4h[i]) or np.isnan(s3_aligned_4h[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(adx_aligned_1d[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned_4h[i]
        r3_val = r3_aligned_4h[i]
        s3_val = s3_aligned_4h[i]
        vol_ma_20_val = vol_ma_20_4h_aligned[i]
        adx_val = adx_aligned_1d[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period 4h average
        vol_filter = vol > 1.8 * vol_ma_20_val if vol_ma_20_val > 0 else False
        
        # Trend filter: 1d ADX > 20 (trending regime)
        trend_filter = adx_val > 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point
            if price <= pp_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point
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
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S3 with volume and trend confirmation
            elif price < s3_val and vol_filter and trend_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_CamarillaR3S3_4hVolumeSpike_1dADXTrend_V1"
timeframe = "1h"
leverage = 1.0