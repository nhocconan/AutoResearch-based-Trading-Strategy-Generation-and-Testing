#!/usr/bin/env python3
# 4h_1d_trix_volume_regime_v2
# Strategy: 4h TRIX(12) momentum + volume spike + TRIX zero-cross for trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: TRIX (Triple Exponential Average) filters momentum in both bull/bear markets.
# Long when TRIX crosses above zero with volume spike; short when crosses below zero with volume spike.
# Uses 1-day TRIX as higher timeframe trend filter to avoid counter-trend trades.
# Targets 30-60 trades/year to minimize fee drag while capturing momentum turns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v2"
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
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d TRIX for trend filter (15-period EMA applied 3x)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).mean()
    trix_1d = 100 * (pd.Series(ema3).pct_change()).values
    trix_1d_smoothed = pd.Series(trix_1d).ewm(span=9, adjust=False).mean().values  # Signal line
    trix_1d_signal = align_htf_to_ltf(prices, df_1d, trix_1d_smoothed)
    
    # 4h TRIX for entry signal
    ema1_4h = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2_4h = pd.Series(ema1_4h).ewm(span=12, adjust=False).mean()
    ema3_4h = pd.Series(ema2_4h).ewm(span=12, adjust=False).mean()
    trix_raw = 100 * (pd.Series(ema3_4h).pct_change()).values
    trix_4h = pd.Series(trix_raw).ewm(span=9, adjust=False).mean().values  # Signal line
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_4h[i]) or np.isnan(trix_1d_signal[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix_4h[i] > 0 and trix_4h[i-1] <= 0  # Cross above zero
        trix_cross_down = trix_4h[i] < 0 and trix_4h[i-1] >= 0  # Cross below zero
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Trend filter: 1d TRIX signal direction
        uptrend_1d = trix_1d_signal[i] > 0
        downtrend_1d = trix_1d_signal[i] < 0
        
        # Long: TRIX crosses above zero with volume in uptrend (1d)
        long_signal = trix_cross_up and vol_confirmed and uptrend_1d
        
        # Short: TRIX crosses below zero with volume in downtrend (1d)
        short_signal = trix_cross_down and vol_confirmed and downtrend_1d
        
        # Exit when TRIX crosses back through zero (mean reversion)
        exit_long = position == 1 and trix_4h[i] < 0
        exit_short = position == -1 and trix_4h[i] > 0
        
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