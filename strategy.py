#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume confirmation and 1d EMA200 trend filter
# Long when price breaks above 1h Camarilla R3 + volume > 1.3x 20-period avg + above 1d EMA200
# Short when price breaks below 1h Camarilla S3 + volume > 1.3x 20-period avg + below 1d EMA200
# Uses Camarilla levels for precise intraday support/resistance, volume for confirmation,
# and 1d EMA200 for regime filter. Designed for moderate trade frequency (20-40/year) with
# session filter (08-20 UTC) to avoid low-liquidity hours. Works in bull/bear via trend alignment.

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
    
    # Get 1h HTF data once before loop (same as primary timeframe for Camarilla calc)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # === 1h Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate daily pivot from previous 1h session (24h lookback for Camarilla)
    lookback = 24  # 24 * 1h = previous day
    if len(high_1h) >= lookback and len(low_1h) >= lookback and len(close_1h) >= lookback:
        # Use rolling window for daily high/low/close
        daily_high = pd.Series(high_1h).rolling(window=lookback, min_periods=lookback).max().shift(1).values
        daily_low = pd.Series(low_1h).rolling(window=lookback, min_periods=lookback).min().shift(1).values
        daily_close = pd.Series(close_1h).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
        
        # Camarilla formulas
        pivot = (daily_high + daily_low + daily_close) / 3
        range_hl = daily_high - daily_low
        r3 = pivot + (range_hl * 1.1 / 2)
        s3 = pivot - (range_hl * 1.1 / 2)
        
        r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 4h Indicator: Volume Confirmation ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
    
    # === 1d Indicator: EMA200 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.3x 20-period volume SMA (aligned to 1h)
        if np.isnan(vol_sma_20_4h_aligned[i]) or volume[i] <= (vol_sma_20_4h_aligned[i] * 1.3):
            vol_confirm = False
        else:
            vol_confirm = True
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1h Camarilla R3
        # 2. Above 1d EMA200 (bullish regime)
        # 3. Volume confirmation
        if (close[i] > r3_aligned[i]) and \
           (close[i] > ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1h Camarilla S3
        # 2. Below 1d EMA200 (bearish regime)
        # 3. Volume confirmation
        elif (close[i] < s3_aligned[i]) and \
             (close[i] < ema_200_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_Volume_4hConfirm_1dEMA200_v1"
timeframe = "1h"
leverage = 1.0