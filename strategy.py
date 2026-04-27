#!/usr/bin/env python3
"""
4h_RSI2_MeanReversion_1dTrend_Filter
Hypothesis: RSI(2) extreme reversals in the direction of daily trend capture mean-reversion bounces during pullbacks in trending markets. Works in both bull and bear markets by only taking long positions in uptrends and short positions in downtrends. Uses RSI(2) for timely entries and daily EMA34 for trend filter to avoid counter-trend trades. Targets 20-50 trades/year on 4h to minimize fee drag while capturing high-probability reversals.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily trend filter: EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # RSI(2) for mean reversion signals
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and trend
    start_idx = max(34, rsi_period + 10)
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_trend = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: RSI(2) oversold in uptrend
            if rsi_val < 10 and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI(2) overbought in downtrend
            elif rsi_val > 90 and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or trend breaks
            if rsi_val > 50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral or trend breaks
            if rsi_val < 50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI2_MeanReversion_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0