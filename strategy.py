#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX regime filter + volume confirmation
# Long when Williams %R(14) crosses above -50 (bullish momentum) + 1d ADX > 25 (trending market) + volume > 1.5x 20-period avg
# Short when Williams %R(14) crosses below -50 (bearish momentum) + 1d ADX > 25 (trending market) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Williams %R identifies overbought/oversold conditions and momentum shifts.
# 1d ADX filter ensures we only trade in trending markets (avoids chop/range).
# Works in bull markets (buying momentum in uptrend) and bear markets (selling momentum in downtrend).

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
    
    # === 1d Indicator: ADX(14) (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    period_adx = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[period_adx-1] = np.mean(tr[:period_adx])
    dm_plus_smooth[period_adx-1] = np.mean(dm_plus[:period_adx])
    dm_minus_smooth[period_adx-1] = np.mean(dm_minus[:period_adx])
    
    # Wilder's smoothing
    for i in range(period_adx, len(tr)):
        atr[i] = (atr[i-1] * (period_adx - 1) + tr[i]) / period_adx
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period_adx - 1) + dm_plus[i]) / period_adx
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period_adx - 1) + dm_minus[i]) / period_adx
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[2*period_adx-1] = np.mean(dx[period_adx-1:2*period_adx])
    for i in range(2*period_adx, len(dx)):
        adx[i] = (adx[i-1] * (period_adx - 1) + dx[i]) / period_adx
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Williams %R(14) ===
    period_willr = 14
    highest_high = pd.Series(high).rolling(window=period_willr, min_periods=period_willr).max().values
    lowest_low = pd.Series(low).rolling(window=period_willr, min_periods=period_willr).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period_willr, 2*period_adx) + 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -50 (bullish momentum)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        if (willr[i] > -50) and (willr[i-1] <= -50) and \
           (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -50 (bearish momentum)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation
        elif (willr[i] < -50) and (willr[i-1] >= -50) and \
             (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_1dADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0