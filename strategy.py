#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_Filter
Hypothesis: Use 4h Donchian channel breakout with volume confirmation and 1d EMA trend filter. Enter long when price breaks above 20-period upper band with volume > 1.5x average and price > 1d EMA50. Enter short when price breaks below lower band with volume confirmation and price < 1d EMA50. Exit when price returns to the middle of the channel (20-period SMA). Designed for 20-40 trades/year to avoid fee drag and work in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Middle band: 20-period SMA
    middle_band = pd.Series(close_4h := df_4h['close'].values).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 4h
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_4h, middle_band)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        if position == 0:
            # Long: price breaks above upper band with volume and above 1d EMA50
            if close[i] > upper_band_aligned[i] and vol_confirm and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume and below 1d EMA50
            elif close[i] < lower_band_aligned[i] and vol_confirm and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band
            if close[i] < middle_band_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band
            if close[i] > middle_band_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0