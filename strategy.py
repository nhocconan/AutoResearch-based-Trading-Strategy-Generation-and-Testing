#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX momentum with 1w EMA50 trend filter and volume spike confirmation.
- Uses TRIX(12,9) as momentum oscillator (long when TRIX > signal line and rising, short when opposite)
- 1w EMA50 as trend filter (long only above, short only below) - avoids counter-trend whipsaw
- Volume > 2.0x 30-period average for confirmation (adjusts for 12h lower frequency)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + momentum confirmation
- Uses 1w HTF as specified in experiment parameters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 30-period average (adjusted for 12h lower frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate TRIX(12,9) on 12h close
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (pd.Series(ema3).pct_change()).values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 12*3, 50, 9)  # Volume MA, TRIX calculation, EMA50, signal line
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(trix[i]) or
            np.isnan(trix_signal[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # TRIX signals
        trix_bullish = trix[i] > trix_signal[i] and trix[i] > trix[i-1]  # Above signal and rising
        trix_bearish = trix[i] < trix_signal[i] and trix[i] < trix[i-1]  # Below signal and falling
        
        if position == 0:
            # Long: TRIX bullish crossover AND price above 1w EMA50 AND volume confirmation
            if trix_bullish and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX bearish crossover AND price below 1w EMA50 AND volume confirmation
            elif trix_bearish and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX bearish crossover OR price crosses below 1w EMA50
            if trix_bearish or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX bullish crossover OR price crosses above 1w EMA50
            if trix_bullish or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_Momentum_1wEMA50_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0