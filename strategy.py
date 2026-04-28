#!/usr/bin/env python3
"""
1d_TRIX_VolumeSpike_1wTrend
Hypothesis: TRIX (triple exponential average) momentum combined with volume spikes and weekly trend filter captures medium-term momentum in both bull and bear markets. TRIX filters out insignificant price movements, volume confirms institutional participation, and weekly trend alignment reduces counter-trend whipsaw. Target: 15-25 trades/year to minimize fee decay while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = pd.Series(ema3).pct_change(1).values * 100  # Percentage change
    
    # Get weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma_30 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_raw[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # TRIX signal: positive = bullish momentum, negative = bearish momentum
        trix_bullish = trix_raw[i] > 0.0
        trix_bearish = trix_raw[i] < 0.0
        
        # Trend filter from weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions with volume confirmation and trend alignment
        long_entry = trix_bullish and volume_spike[i] and uptrend
        short_entry = trix_bearish and volume_spike[i] and downtrend
        
        # Exit on opposite TRIX signal
        long_exit = trix_bearish and volume_spike[i]
        short_exit = trix_bullish and volume_spike[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0