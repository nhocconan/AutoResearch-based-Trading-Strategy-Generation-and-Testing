#!/usr/bin/env python3
name = "4h_Bollinger_Squeeze_Momentum_200EMA"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for Bollinger Bands and 200 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Bollinger Bands (20, 2.0)
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    
    # Daily 200 EMA for trend filter
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # 4h Bollinger Band width for squeeze detection
    sma20_4h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20_4h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb_4h = sma20_4h + 2.0 * std20_4h
    lower_bb_4h = sma20_4h - 2.0 * std20_4h
    bb_width = (upper_bb_4h - lower_bb_4h) / sma20_4h
    
    # Bollinger Band width percentile (20-period) for squeeze
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # 4h volume confirmation (above 1.5x 20-period average)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze: low volatility percentile
            squeeze_condition = bb_width_percentile[i] < 0.2
            
            # Long: Price breaks above upper BB with volume, during squeeze, above daily 200 EMA
            if (squeeze_condition and close[i] > upper_bb_4h[i] and vol_confirm[i] and 
                close[i] > ema200_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB with volume, during squeeze, below daily 200 EMA
            elif (squeeze_condition and close[i] < lower_bb_4h[i] and vol_confirm[i] and 
                  close[i] < ema200_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns to middle BB or volatility expands
            middle_bb_4h = sma20_4h[i]
            if close[i] < middle_bb_4h or bb_width_percentile[i] > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to middle BB or volatility expands
            middle_bb_4h = sma20_4h[i]
            if close[i] > middle_bb_4h or bb_width_percentile[i] > 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Bollinger squeeze strategy: enters during low volatility breakouts with volume confirmation
# Uses daily 200 EMA for trend filter and daily Bollinger Bands for dynamic S/R
# Position size 0.25 limits risk. Target ~20-40 trades/year to minimize fee drag.
# Exit on return to middle Bollinger Band or volatility expansion.