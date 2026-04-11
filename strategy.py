#!/usr/bin/env python3
"""
4h_1d_trix_volume_reversal
Strategy: 4h TRIX reversal with volume spike and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Buy when TRIX crosses above zero with volume spike in uptrend (1d close > prior); sell when TRIX crosses below zero with volume spike in downtrend (1d close < prior). Uses volume confirmation to avoid false signals and trend filter to avoid counter-trend trades. Low-frequency design targets 20-50 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_reversal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX: 12-period EMA of 12-period EMA of 12-period EMA of close, then ROC
    # EMA1
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX: 1-period percent change of EMA3
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Histogram: TRIX - signal
    trix_hist = trix_raw - trix_signal
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d trend: today's close vs yesterday's close
    close_1d = df_1d['close'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_prev)
    
    # Volume filter: 4h volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_hist[i]) or np.isnan(trix_signal[i]) or
            np.isnan(close_1d_trend[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        trix = trix_hist[i]
        trix_prev = trix_hist[i-1] if i > 0 else 0
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Trend filter: 1d close vs prior day close
        uptrend_1d = close[i] > close_1d_trend[i]
        downtrend_1d = close[i] < close_1d_trend[i]
        
        # Long conditions: TRIX crosses above zero with volume + 1d uptrend
        long_signal = volume_confirmed and (trix_prev <= 0) and (trix > 0) and uptrend_1d
        
        # Short conditions: TRIX crosses below zero with volume + 1d downtrend
        short_signal = volume_confirmed and (trix_prev >= 0) and (trix < 0) and downtrend_1d
        
        # Exit when TRIX crosses back through zero (mean reversion)
        exit_long = position == 1 and (trix_prev > 0) and (trix <= 0)
        exit_short = position == -1 and (trix_prev < 0) and (trix >= 0)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Buy when TRIX crosses above zero with volume spike in uptrend (1d close > prior); sell when TRIX crosses below zero with volume spike in downtrend (1d close < prior). Uses volume confirmation to avoid false signals and trend filter to avoid counter-trend trades. Low-frequency design targets 20-50 trades/year to minimize fee drift.