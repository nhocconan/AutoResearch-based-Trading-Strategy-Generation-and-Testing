#!/usr/bin/env python3
"""
12h_1d_trix_volume_crossover_v1
Strategy: 12h TRIX crossover with volume confirmation and 1d trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses TRIX (triple smoothed EMA) momentum on 12h timeframe for entry signals, confirmed by volume spikes (>2x average) and filtered by 1d EMA50 trend alignment. TRIX captures momentum changes early while reducing whipsaw. Volume confirmation ensures institutional interest. Trend filter avoids counter-trend trades. Designed for low frequency (15-35 trades/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_crossover_v1"
timeframe = "12h"
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
    
    # 12h TRIX (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_values = trix.values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_values[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX crossover signals
        trix_cross_above = trix_values[i] > trix_signal[i] and trix_values[i-1] <= trix_signal[i-1]
        trix_cross_below = trix_values[i] < trix_signal[i] and trix_values[i-1] >= trix_signal[i-1]
        
        # Trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: TRIX bullish crossover with volume in uptrend
        long_signal = trix_cross_above and vol_confirmed and uptrend
        
        # Short: TRIX bearish crossover with volume in downtrend
        short_signal = trix_cross_below and vol_confirmed and downtrend
        
        # Exit when TRIX crosses back in opposite direction
        exit_long = position == 1 and trix_values[i] < trix_signal[i]
        exit_short = position == -1 and trix_values[i] > trix_signal[i]
        
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