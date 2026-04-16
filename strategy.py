#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume spike and 1d ADX trend filter
# Long when price breaks above Camarilla R3 level + volume > 1.5x 20-period average + 1d ADX > 25
# Short when price breaks below Camarilla S3 level + volume > 1.5x 20-period average + 1d ADX > 25
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla pivots provide mathematically derived support/resistance levels that work in ranging markets.
# Volume confirmation ensures breakouts have conviction. ADX filter avoids sideways chop.
# Target: 12-35 trades/year on 6h timeframe to stay within fee-efficient range.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 12h Indicator: Volume SMA (20-period) ===
    vol_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 1d Indicator: ADX (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0  # Handle division by zero
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Camarilla Pivot Levels (from previous day) ===
    # Typical Price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_r3 = typical_price + (range_1d * 1.1 / 4)
    camarilla_s3 = typical_price - (range_1d * 1.1 / 4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 14) + 5  # Volume(20) + ADX(14) + Camarilla + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 12h volume SMA (20-period)
        vol_confirm = volume[i] > (vol_sma_20_12h_aligned[i] * 1.5)
        
        # ADX filter: trending market (ADX > 25)
        adx_filter = adx_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 level
        # 2. Volume confirmation
        # 3. ADX trend filter
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm and adx_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 level
        # 2. Volume confirmation
        # 3. ADX trend filter
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm and adx_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_12hVol_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0