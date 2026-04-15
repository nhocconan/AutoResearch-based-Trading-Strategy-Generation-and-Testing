#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze with Volume Spike and ATR Stop
# Identifies low volatility periods (BB width < 20th percentile) followed by volatility expansion
# (BB width > 80th percentile) with volume confirmation (>2x average volume).
# Trades in direction of breakout (close outside Bollinger Bands).
# Works in both bull and bear markets by capturing volatility expansion after contraction.
# Uses 1d ATR for stop loss and 1w trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on 1d
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Bollinger Band width percentile (20th and 80th) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_20th = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_width_80th = bb_width_series.rolling(window=50, min_periods=50).quantile(0.80).values
    
    # Squeeze condition: BB width < 20th percentile (low volatility)
    squeeze = bb_width < bb_width_20th
    
    # Expansion condition: BB width > 80th percentile (high volatility)
    expansion = bb_width > bb_width_80th
    
    # Breakout conditions
    breakout_up = close_1d > upper_bb
    breakout_down = close_1d < lower_bb
    
    # Volume condition: volume > 2x average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2 * avg_volume
    
    # Trend filter: 50-period EMA on 1w
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1d indicators to 12h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    expansion_aligned = align_htf_to_ltf(prices, df_1d, expansion)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # ATR for stop loss (14-period on 1d)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(expansion_aligned[i]) or
            np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(uptrend_1w_aligned[i]) or
            np.isnan(downtrend_1w_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            continue
        
        # Long entry: bullish breakout after squeeze, with volume spike and uptrend
        if (squeeze_aligned[i-1] and expansion_aligned[i] and breakout_up_aligned[i] and
            volume_spike_aligned[i] and uptrend_1w_aligned[i] and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bearish breakout after squeeze, with volume spike and downtrend
        elif (squeeze_aligned[i-1] and expansion_aligned[i] and breakout_down_aligned[i] and
              volume_spike_aligned[i] and downtrend_1w_aligned[i] and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or volatility contraction (squeeze returns)
        elif position == 1 and (breakout_down_aligned[i] or squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up_aligned[i] or squeeze_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Spike"
timeframe = "12h"
leverage = 1.0