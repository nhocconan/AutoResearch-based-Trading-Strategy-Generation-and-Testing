#!/usr/bin/env python3
"""
Hypothesis: Daily Bollinger Band Squeeze Breakout with Weekly Trend Filter and Volume Confirmation.
Buy when price breaks above upper Bollinger Band during low volatility (squeeze) and weekly trend is up.
Sell when price breaks below lower Bollinger Band during low volatility and weekly trend is down.
Uses Bollinger Band width to detect low volatility regimes, with breakouts capturing explosive moves.
Works in both bull and bear markets by following the weekly trend direction, avoiding counter-trend trades.
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
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band Width for squeeze detection (low volatility)
    bb_width = (upper - lower) / basis
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < 0.5 * bb_width_ma  # Width less than 50% of its MA
    
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
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(squeeze_condition[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper BB during squeeze and weekly uptrend
            if close[i] > upper[i] and squeeze_condition[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB during squeeze and weekly downtrend
            elif close[i] < lower[i] and squeeze_condition[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle band (mean reversion) or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below basis or weekly trend turns down
                if close[i] < basis[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above basis or weekly trend turns up
                if close[i] > basis[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
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