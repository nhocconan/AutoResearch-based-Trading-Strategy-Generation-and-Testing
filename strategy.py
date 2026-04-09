#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and choppiness regime filter on 1d HTF.
# Long when price breaks above 20-period Donchian high with volume > 1.5x 20-period average AND 1d chop > 61.8 (range).
# Short when price breaks below 20-period Donchian low with volume > 1.5x 20-period average AND 1d chop > 61.8 (range).
# Exit when price closes back inside the Donchian channel (mean reversion in ranging markets).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channel (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Get daily data for choppiness filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for chop calculation
        return np.zeros(n)
    
    # Calculate daily choppiness index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with close_1d index
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_14 / range_14) / np.log10(14)
    
    # Align chop to 12h timeframe (wait for completed 1d bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: chop > 61.8 indicates ranging market (good for mean reversion breakouts)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price closes back inside Donchian channel (mean reversion)
            if close[i] <= donchian_high[i] and close[i] >= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes back inside Donchian channel (mean reversion)
            if close[i] <= donchian_high[i] and close[i] >= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume and chop confirmation
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and chop_filter
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and chop_filter
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals