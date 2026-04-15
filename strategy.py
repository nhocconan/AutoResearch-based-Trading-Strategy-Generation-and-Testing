#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w ADX regime filter
# Long when price breaks above 1d Camarilla R3 + volume > 2x 20-period avg + 1w ADX > 25 (trending)
# Short when price breaks below 1d Camarilla S3 + volume > 2x 20-period avg + 1w ADX > 25 (trending)
# Uses discrete position sizing (0.25) to reduce fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels from 1d provide intraday support/resistance. Volume confirms breakout strength.
# 1w ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges.

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
    
    # Get 1d HTF data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    S3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    
    # Align to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # === 1w Indicator: ADX (trend regime filter) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period_adx = 14
    atr_1w = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = np.where(atr_1w != 0, (dm_plus_smooth / atr_1w) * 100, 0)
    di_minus = np.where(atr_1w != 0, (dm_minus_smooth / atr_1w) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, period_adx)
    
    # Align to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + period_adx  # 1d lookback + volume + ADX
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1w ADX > 25 (trending market)
        trending_regime = adx_1w_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3
        # 2. Volume confirmation
        # 3. Trending regime (ADX > 25)
        if (close[i] > R3_1d_aligned[i]) and \
           vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3
        # 2. Volume confirmation
        # 3. Trending regime (ADX > 25)
        elif (close[i] < S3_1d_aligned[i]) and \
             vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolume_1wADX_Regime_v1"
timeframe = "12h"
leverage = 1.0