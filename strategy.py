#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_trix_volume_signal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate TRIX on daily close
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1-period ago value, then % change
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = (ema3 - ema3_previous) / ema3_previous * 100
    trix_raw = np.diff(ema3, prepend=ema3[0])
    trix = (trix_raw / ema3) * 100
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: 12h volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.3 * vol_ma_20[i]
        
        # TRIX signal: positive = bullish momentum, negative = bearish momentum
        trix_signal = trix_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TRIX > 0 (bullish momentum) + volume confirmation
        if trix_signal > 0 and vol_confirm:
            enter_long = True
        
        # Short: TRIX < 0 (bearish momentum) + volume confirmation
        if trix_signal < 0 and vol_confirm:
            enter_short = True
        
        # Exit conditions: TRIX crosses zero
        exit_long = trix_signal <= 0
        exit_short = trix_signal >= 0
        
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

# Hypothesis: TRIX (Triple Exponential Moving Average) on daily timeframe measures momentum
# by showing the rate of change of a triple-smoothed EMA. Positive TRIX indicates bullish
# momentum, negative indicates bearish momentum. Combined with volume confirmation on 12h
# timeframe to ensure participation. Works in both bull and bear markets by following
# momentum direction. Position size 0.25 limits drawdown. Target: 50-150 trades over 4 years.