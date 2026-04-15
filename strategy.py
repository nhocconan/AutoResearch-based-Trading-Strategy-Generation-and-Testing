#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA21 trend filter and volume spike
# Long when price breaks above Camarilla R3 + 4h EMA21 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S3 + 4h EMA21 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# Camarilla levels provide intraday support/resistance that work in ranging and trending markets.
# 4h EMA21 provides medium-term trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-30 trades/year on 1h timeframe to avoid overtrading.
# Session filter (08-20 UTC) focuses on high-liquidity periods.

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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA21 ===
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 1h Camarilla Pivot Points (based on previous day) ===
    # Typical Price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Daily OHLC from 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point = (prev_high + prev_low + prev_close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Camarilla levels
    # R3 = pivot + (prev_high - prev_low) * 1.1 / 4
    # S3 = pivot - (prev_high - prev_low) * 1.1 / 4
    camarilla_r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe (they change daily at 00:00 UTC)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(21, 20) + 5  # EMA21 + Camarilla + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. 4h EMA21 uptrend (close > EMA21)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i]) and \
           (close[i] > ema_21_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. 4h EMA21 downtrend (close < EMA21)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i]) and \
             (close[i] < ema_21_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_4hEMA21_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0