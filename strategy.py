#!/usr/bin/env python3
"""
Hypothesis: Daily Bollinger Band Squeeze Breakout with Weekly Trend Filter and Volume Confirmation.
Long when price breaks above upper Bollinger Band during low volatility (squeeze) and weekly trend is up.
Short when price breaks below lower Bollinger Band during low volatility (squeeze) and weekly trend is down.
Exit when price returns to middle Bollinger Band or weekly trend reverses.
Bollinger Band squeeze indicates low volatility pre-breakout; breakout captures the volatility expansion.
Weekly trend filter ensures alignment with higher-timeframe momentum. Works in both bull and bear markets
by following the weekly trend direction, avoiding counter-trend trades during strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / basis
    bb_width_ma = bb_width.rolling(window=50, min_periods=50).mean()
    squeeze = bb_width < bb_width_ma  # True when volatility is low relative to average
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(basis.iloc[i]) or np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper BB during squeeze, weekly trend up, volume spike
            if (close[i] > upper.iloc[i] and squeeze.iloc[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB during squeeze, weekly trend down, volume spike
            elif (close[i] < lower.iloc[i] and squeeze.iloc[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle BB or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to basis or weekly trend turns down
                if close[i] <= basis.iloc[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to basis or weekly trend turns up
                if close[i] >= basis.iloc[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_BollingerBand_Squeeze_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0