#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Squeeze + 1d Momentum Breakout
# In low volatility (Bollinger Bandwidth < 50th percentile), price consolidates.
# A breakout with volume confirmation and aligned 1d momentum (price > 1d EMA50) captures the move.
# Works in bull/bear: squeeze identifies compression, breakout direction follows momentum.
# Targets 20-40 trades/year (~80-160 total) to minimize fee drag.

name = "6h_BollingerSqueeze_1dEMA50_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    
    # Bollinger Bandwidth: (upper - lower) / basis
    bw = (upper - lower) / basis
    # Squeeze threshold: 50th percentile of bandwidth (lookback 50 periods)
    bw_percentile = pd.Series(bw).rolling(window=50, min_periods=30).quantile(0.50).values
    squeeze = bw < bw_percentile
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bw_percentile[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: squeeze breakout above upper band, price > 1d EMA50, volume confirmation
            if close[i] > upper[i] and squeeze[i] and close[i] > ema50_1d_aligned[i] and vol_conf[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout below lower band, price < 1d EMA50, volume confirmation
            elif close[i] < lower[i] and squeeze[i] and close[i] < ema50_1d_aligned[i] and vol_conf[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below basis OR squeeze breaks down (bandwidth expands)
            if close[i] < basis[i] or bw[i] > bw_percentile[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above basis OR squeeze breaks down
            if close[i] > basis[i] or bw[i] > bw_percentile[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals