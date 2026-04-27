#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ATR-based volatility breakout with 1d trend filter and volume confirmation
# Uses ATR(14) to detect volatility expansion: long when price breaks above close + 0.5*ATR,
# short when breaks below close - 0.5*ATR. Requires 1d EMA50 trend alignment to avoid whipsaws.
# Volume confirmation: current volume > 1.5x 20-period average.
# Designed for 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # ATR(14) calculation on 4h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 50-period EMA on daily close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility breakout levels
        upper_break = close[i-1] + 0.5 * atr[i]
        lower_break = close[i-1] - 0.5 * atr[i]
        
        # Long conditions: price breaks above upper band AND 1d uptrend AND volume
        if (close[i] > upper_break and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below lower band AND 1d downtrend AND volume
        elif (close[i] < lower_break and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_ATRBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0