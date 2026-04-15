#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Long when price breaks above 1d Camarilla R3 level + 1d ADX > 20 + volume > 2x 20-period avg
# Short when price breaks below 1d Camarilla S3 level + 1d ADX > 20 + volume > 2x 20-period avg
# Camarilla levels provide intraday support/resistance based on prior day's range.
# ADX filter ensures we trade in trending markets, avoiding chop. Volume spike confirms conviction.
# Designed for low trade frequency (12-30/year) with discrete position sizing (0.25) to minimize fee drag.
# Works in bull markets (breakout continuation) and bear markets (strong downtrends) by requiring ADX > 20.

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
    
    # === 1d Indicator: ADX (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    # Based on prior day's OHLC
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    
    # Handle first bar
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    close_1d_shift[0] = close_1d[0]
    
    # Camarilla calculations
    camarilla_high = high_1d_shift
    camarilla_low = low_1d_shift
    camarilla_close = close_1d_shift
    
    camarilla_range = camarilla_high - camarilla_low
    
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_r3 = camarilla_close + camarilla_range * 1.1 / 4
    camarilla_s3 = camarilla_close - camarilla_range * 1.1 / 4
    
    # Align to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(2*period, 20) + 5  # ADX(28) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Trend (1d ADX > 20)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and \
           (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Trend (1d ADX > 20)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and \
             (adx_aligned[i] > 20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_CamarillaR3S3_1dADX20_Volume2x_Filter_v1"
timeframe = "12h"
leverage = 1.0