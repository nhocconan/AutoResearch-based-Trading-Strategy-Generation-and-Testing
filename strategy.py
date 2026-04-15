#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# with 1d volume confirmation and 1d ADX regime filter. In ranging markets (ADX<25), fade extreme
# Camarilla levels (R3/S3). In trending markets (ADX>25), breakout continuation at R4/S4.
# Designed for low trade frequency (12-30/year) to minimize fee drag while adapting to market regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    open_time = prices['open_time'].values
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    # Calculate Camarilla levels
    rang_12h = high_12h - low_12h
    r3_12h = close_12h + rang_12h * 1.1 / 4.0
    s3_12h = close_12h - rang_12h * 1.1 / 4.0
    r4_12h = close_12h + rang_12h * 1.1 / 2.0
    s4_12h = close_12h - rang_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 1d Indicators: Volume and ADX ===
    # Volume confirmation: current volume > 1.5x 20-period volume SMA
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = df_1d['volume'].values > (vol_sma_20 * 1.5)
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    # ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_confirm_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        if not vol_confirm_1d_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Regime detection: ADX > 25 = trending, ADX < 25 = ranging
        is_trending = adx_aligned[i] > 25
        
        if is_trending:
            # TRENDING MARKET: Breakout continuation at R4/S4
            if close[i] > r4_12h_aligned[i]:
                signals[i] = 0.25  # Long breakout
            elif close[i] < s4_12h_aligned[i]:
                signals[i] = -0.25  # Short breakout
            else:
                signals[i] = 0.0
        else:
            # RANGING MARKET: Mean reversion at R3/S3
            if close[i] > r3_12h_aligned[i]:
                signals[i] = -0.25  # Short at resistance
            elif close[i] < s3_12h_aligned[i]:
                signals[i] = 0.25   # Long at support
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_1dVol_ADX_Regime_v1"
timeframe = "6h"
leverage = 1.0