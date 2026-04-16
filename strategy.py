#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume confirmation and weekly ADX regime filter.
# Long when price breaks above Camarilla R3 (1d) AND 1d volume > 1.5x 20-period average AND weekly ADX < 25 (range/low trend).
# Short when price breaks below Camarilla S3 (1d) AND 1d volume > 1.5x 20-period average AND weekly ADX < 25.
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance, volume confirms participation,
# weekly ADX ensures we only trade in low-volatility regimes to avoid whipsaws. Designed to capture mean-reversion
# bounces off key levels in ranging markets while avoiding strong trends.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Camarilla calculation and volume MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla R3 and S3 levels ===
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    # Using previous day's values to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + 1.1 * camarilla_range / 2
    camarilla_s3 = prev_close_1d - 1.1 * camarilla_range / 2
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get weekly data once before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: ADX (14-period) for regime filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First TR is undefined
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    # First values are undefined
    dm_plus[0] = np.nan
    dm_minus[0] = np.nan
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        """Calculate Wilder's smoothing (EMA with alpha=1/period)"""
        return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    period = 14
    atr_1w = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / np.where(atr_1w == 0, np.nan, atr_1w)
    di_minus = 100 * dm_minus_smoothed / np.where(atr_1w == 0, np.nan, atr_1w)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx_1w = wilders_smoothing(dx, period)
    
    # Align weekly ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Enough for all indicators to stabilize
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        adx = adx_1w_aligned[i]
        
        # Regime filter: only trade when ADX < 25 (low trend/range market)
        in_range_regime = adx < 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls back below R3 or volume spike ends or regime changes
            if price < r3_level or not vol_spike or not in_range_regime:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises back above S3 or volume spike ends or regime changes
            if price > s3_level or not vol_spike or not in_range_regime:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and in_range_regime:
            # LONG: Price breaks above Camarilla R3 AND volume spike
            if price > r3_level and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S3 AND volume spike
            elif price < s3_level and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dVolumeSpike_1wADXFilter_V1"
timeframe = "12h"
leverage = 1.0