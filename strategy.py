#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (SMMA jaws/teeth/lips) with 1d trend filter and volume spike
# Alligator identifies trend phases: jaws (13), teeth (8), lips (5) SMMA.
# - Trending: lips > teeth > jaws (bull) or lips < teeth < jaws (bear)
# - Sleeping: intertwined lines (no trade)
# - Awakening: separation after sleep (trend start)
# Combined with 1d EMA50 trend filter and volume spike for confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # SmMA(t) = (SmMA(t-1) * (period-1) + close(t)) / period
    close_1d = pd.Series(df_1d['close'].values)
    
    # Jaws: SMMA(13)
    smma13 = np.full(len(close_1d), np.nan)
    smma13[12] = close_1d.iloc[:13].mean()
    for i in range(13, len(close_1d)):
        smma13[i] = (smma13[i-1] * 12 + close_1d.iloc[i]) / 13
    
    # Teeth: SMMA(8)
    smma8 = np.full(len(close_1d), np.nan)
    smma8[7] = close_1d.iloc[:8].mean()
    for i in range(8, len(close_1d)):
        smma8[i] = (smma8[i-1] * 7 + close_1d.iloc[i]) / 8
    
    # Lips: SMMA(5)
    smma5 = np.full(len(close_1d), np.nan)
    smma5[4] = close_1d.iloc[:5].mean()
    for i in range(5, len(close_1d)):
        smma5[i] = (smma5[i-1] * 4 + close_1d.iloc[i]) / 5
    
    # Align to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, smma13)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, smma8)
    lips_aligned = align_htf_to_ltf(prices, df_1d, smma5)
    
    # 1d EMA50 for trend filter
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
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
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: lips > teeth > jaws = bullish alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaws_aligned[i])
        # Bearish alignment: lips < teeth < jaws
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaws_aligned[i])
        
        # Long conditions: bullish alignment + price above EMA50 (uptrend) + volume
        if bullish_alignment and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short conditions: bearish alignment + price below EMA50 (downtrend) + volume
        elif bearish_alignment and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_WilliamsAlligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0