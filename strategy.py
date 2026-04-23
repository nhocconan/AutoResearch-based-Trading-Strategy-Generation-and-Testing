#!/usr/bin/env python3
"""
Hypothesis: 12-hour timeframe with 1-day Bollinger Band squeeze breakout + volume confirmation.
Long when price breaks above upper BB after squeeze (BBW < 0.03 for 3 consecutive days),
Short when price breaks below lower BB after squeeze.
Exit when price returns to middle BB or volatility expands (BBW > 0.06).
Designed for low trade frequency (~15-25/year) to capture volatility breakouts in both bull and bear markets.
Uses Bollinger Band width as volatility filter to avoid whipsaws in low volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    
    # Middle band (20-period SMA)
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Standard deviation (20-period)
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    
    # Upper and lower bands
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    
    # Bollinger Band Width (normalized by middle band)
    bbw = (upper_bb - lower_bb) / ma_20
    
    # Squeeze detection: BBW < 0.03 for 3 consecutive days
    squeeze_condition = bbw < 0.03
    squeeze_3d = pd.Series(squeeze_condition).rolling(window=3, min_periods=3).sum() == 3
    squeeze_3d_values = squeeze_3d.values
    
    # Expansion signal: BBW > 0.06 (exit condition)
    expansion_condition = bbw > 0.06
    
    # Align HTF indicators to lower timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_3d_values)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ma_20_aligned = align_htf_to_ltf(prices, df_1d, ma_20)
    expansion_aligned = align_htf_to_ltf(prices, df_1d, expansion_condition)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(ma_20_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        squeeze_val = squeeze_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        ma_20_val = ma_20_aligned[i]
        expansion_val = expansion_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: BB breakout above upper band after squeeze, volume confirmation
            if (squeeze_val and close[i] > upper_bb_val and 
                vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: BB breakout below lower band after squeeze, volume confirmation
            elif (squeeze_val and close[i] < lower_bb_val and 
                  vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle BB OR volatility expansion
                if close[i] <= ma_20_val or expansion_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle BB OR volatility expansion
                if close[i] >= ma_20_val or expansion_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_BollingerSqueeze_Breakout_Volume"
timeframe = "12h"
leverage = 1.0