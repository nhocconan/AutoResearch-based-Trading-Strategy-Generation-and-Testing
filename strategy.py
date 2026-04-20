#!/usr/bin/env python3
# Strategy: 12h_1d_TRIX_VolumeSpike_TrendFilter_v1
# Hypothesis: TRIX (triple EMA) momentum with volume confirmation and 1d EMA34 trend filter on 12h timeframe.
# Long when TRIX crosses above zero with volume > 2x 20-period MA and price above 1d EMA34 (uptrend).
# Short when TRIX crosses below zero with volume confirmation and price below 1d EMA34 (downtrend).
# Uses TRIX to capture momentum shifts and volume to confirm institutional participation.
# Trend filter prevents counter-trend trades. Designed for 15-35 trades/year to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 12h data for TRIX, volume
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # TRIX calculation (triple EMA of 12-period)
    close_12h_series = pd.Series(close_12h)
    ema1 = close_12h_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_values = trix.values
    
    # Volume spike detection (20-period on 12h)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # ATR for volatility filter (14-period on 12h)
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_low[0] = high_12h[0] - low_12h[0]
    high_close[0] = np.abs(high_12h[0] - close_12h[0])
    low_close[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(trix_values[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1] if i > 0 else 0
        
        if position == 0:
            # Long: TRIX crosses above zero, above 1d EMA34 (uptrend), with volume confirmation
            if (trix_now > 0 and trix_prev <= 0 and 
                price > ema34_1d_aligned[i] and 
                vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, below 1d EMA34 (downtrend), with volume confirmation
            elif (trix_now < 0 and trix_prev >= 0 and 
                  price < ema34_1d_aligned[i] and 
                  vol > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below zero or ATR-based stop
            if (trix_now < 0 or 
                price < low_12h[i] - 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero or ATR-based stop
            if (trix_now > 0 or 
                price > high_12h[i] + 1.5 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_TRIX_VolumeSpike_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0