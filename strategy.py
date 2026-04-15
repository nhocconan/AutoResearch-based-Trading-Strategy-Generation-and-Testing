#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Camarilla pivot R3/S3 breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3 + 1d EMA50 uptrend + volume > 1.8x 20-period avg
# Short when price breaks below Camarilla S3 + 1d EMA50 downtrend + volume > 1.8x 20-period avg
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 1d EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.8x) targets ~15-30 trades/year on 1h timeframe to avoid overtrading.
# Camarilla pivots calculated from prior 4h bar's high/low/close for structure-based entries.

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
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Camarilla Pivot Levels (based on prior bar) ===
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1 / 4
    # S3 = Pivot - (H - L) * 1.1 / 4
    # Using prior bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_4h_Camarilla_R3S3_1dEMA50_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0