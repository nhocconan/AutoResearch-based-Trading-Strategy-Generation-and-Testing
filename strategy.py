#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return signals
    
    # Calculate TRIX on daily close (15-period EMA triple)
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value has no previous
    
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: 4h volume > 1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_50[i]
        
        # TRIX momentum conditions
        trix_bullish = trix_aligned[i] > 0
        trix_bearish = trix_aligned[i] < 0
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TRIX positive with volume confirmation
        if trix_bullish and vol_confirm:
            enter_long = True
        
        # Short: TRIX negative with volume confirmation
        if trix_bearish and vol_confirm:
            enter_short = True
        
        # Exit conditions: TRIX crosses zero
        exit_long = trix_aligned[i] <= 0  # TRIX crosses below zero
        exit_short = trix_aligned[i] >= 0  # TRIX crosses above zero
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h TRIX momentum strategy with daily TRIX and volume confirmation.
# Enters long when daily TRIX > 0 (bullish momentum) with volume > 1.5x 50-period average.
# Enters short when daily TRIX < 0 (bearish momentum) with volume > 1.5x 50-period average.
# Exits when TRIX crosses zero, indicating momentum shift.
# Uses volume confirmation to filter false signals and reduce trade frequency.
# Position size 0.25 manages risk in volatile markets.
# Target: 20-30 trades per year (80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by capturing momentum shifts in either direction.
# 4h timeframe balances signal quality with trade frequency.