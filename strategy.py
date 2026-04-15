#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot R3/S3 breakout with 4h ADX trend filter and volume confirmation
# Long when price breaks above 1h Camarilla R3 + 4h ADX > 25 + volume > 1.5x 20-period avg
# Short when price breaks below 1h Camarilla S3 + 4h ADX > 25 + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Camarilla pivots provide mathematically derived support/resistance levels. ADX filter ensures we only trade strong trends.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring ADX > 25.
# 1h timeframe allows precise entry timing while 4h/1d HTF filters reduce noise and overtrading.

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: ADX (trend strength filter) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_4h_shift = np.roll(high_4h, 1)
    low_4h_shift = np.roll(low_4h, 1)
    high_4h_shift[0] = high_4h[0]
    low_4h_shift[0] = low_4h[0]
    
    plus_dm = np.where((high_4h - high_4h_shift) > (low_4h_shift - low_4h), 
                       np.maximum(high_4h - high_4h_shift, 0), 0)
    minus_dm = np.where((low_4h_shift - low_4h) > (high_4h - high_4h_shift), 
                        np.maximum(low_4h_shift - low_4h, 0), 0)
    
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
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
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # === 1h Indicator: Camarilla Pivots (R3, S3) ===
    # Camarilla levels based on previous day's range
    # For intraday, we use rolling window of 24 periods (24*1h = 1 day)
    lookback = 24  # 24 hours = 1 day
    if len(high) < lookback:
        return np.zeros(n)
    
    # Calculate rolling max/min for the lookback period
    rolling_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    rolling_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    rolling_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Camarilla formulas:
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    camarilla_range = rolling_high - rolling_low
    camarilla_r3 = rolling_close + camarilla_range * 1.1 / 4
    camarilla_s3 = rolling_close - camarilla_range * 1.1 / 4
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 2*period) + 20  # Camarilla(24) + ADX(28) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1h Camarilla R3
        # 2. Trend (4h ADX > 25)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1h Camarilla S3
        # 2. Trend (4h ADX > 25)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (adx_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_4hADX25_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0