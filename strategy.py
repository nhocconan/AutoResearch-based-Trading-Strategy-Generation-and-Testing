#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ATR regime filter and volume spike confirmation.
- Bollinger Bands: 20-period SMA, 2.0 std dev
- Squeeze condition: BB width < 50th percentile of last 50 periods (low volatility)
- Breakout: Close breaks above upper band (long) or below lower band (short)
- Regime filter: 1d ATR(14) > 50th percentile of last 100 periods (high volatility regime)
- Volume confirmation: Volume > 2.0x 20-period average
- Works in both bull and breakout bear markets by capturing volatility expansion after consolidation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std * bb_std_dev)
    lower_band = sma_bb - (bb_std * bb_std_dev)
    bb_width = upper_band - lower_band
    
    # BB width percentile (50th = median) over last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.50).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * vol_ma
    
    # 1d ATR regime filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # ATR regime: > 50th percentile of last 100 periods (high volatility)
    atr_1d_series = pd.Series(atr_1d)
    atr_percentile = atr_1d_series.rolling(window=100, min_periods=100).quantile(0.50).values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    high_volatility_regime = atr_1d_aligned > atr_percentile_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_period, 20, 50, 100, 14)  # BB=20, vol=20, bb_width_percentile=50, atr_percentile=100, atr=14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_bb[i]) or 
            np.isnan(bb_width_percentile[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_percentile_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_band[i]
        short_breakout = close[i] < lower_band[i]
        
        if position == 0:
            # Long: Squeeze + breakout up + volume + high volatility regime
            if (squeeze_condition[i] and long_breakout and 
                volume_confirm[i] and 
                high_volatility_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Squeeze + breakout down + volume + high volatility regime
            elif (squeeze_condition[i] and short_breakout and 
                  volume_confirm[i] and 
                  high_volatility_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Breakdown below middle band OR volatility contraction
            if close[i] < sma_bb[i] or not high_volatility_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Breakout above middle band OR volatility contraction
            if close[i] > sma_bb[i] or not high_volatility_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBSqueeze_Breakout_ATRRegime_Volume"
timeframe = "6h"
leverage = 1.0