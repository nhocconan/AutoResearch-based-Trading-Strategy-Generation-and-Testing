#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# Long when price breaks above Camarilla R3 (1d) + 1d volume > 1.5x 20-period average + 1d ADX > 25 (trending market)
# Short when price breaks below Camarilla S3 (1d) + 1d volume > 1.5x 20-period average + 1d ADX > 25 (trending market)
# Uses 1d Camarilla levels for structure, 6h for execution timing, volume for confirmation, ADX to avoid ranging markets
# Designed for low trade frequency (15-25/year) to minimize fee drag in trending markets
# Session filter (08-20 UTC) avoids low-liquidity hours
# Works in both bull and bear markets by requiring trending conditions (ADX > 25) and volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    camarilla_r3_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # === 1d Indicator: Volume Confirmation ===
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values if 'volume_1d' in df_1d.columns else \
                    pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1d Indicator: ADX Trend Filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - close_1d.shift(1)))
    tr3 = pd.Series(abs(low_1d - close_1d.shift(1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Calculate Directional Movement
    dm_plus = pd.Series(high_1d - high_1d.shift(1))
    dm_minus = pd.Series(low_1d.shift(1) - low_1d)
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Calculate smoothed TR and DM
    tr_period = 14
    atr_1d = pd.Series(tr_1d).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / atr_1d)
    di_minus = 100 * (dm_minus_smooth / atr_1d)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Primary: 6h price for breakout detection ===
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period volume SMA
        vol_confirm = df_1d['volume'].iloc[-1] > (vol_sma_20_1d_aligned[i] * 1.5) if len(df_1d) > 0 else False
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_1d_aligned[i] > 25
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Trending market (ADX > 25)
        if (close[i] > camarilla_r3_1d_aligned[i]) and vol_confirm and adx_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Trending market (ADX > 25)
        elif (close[i] < camarilla_s3_1d_aligned[i]) and vol_confirm and adx_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolume_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0