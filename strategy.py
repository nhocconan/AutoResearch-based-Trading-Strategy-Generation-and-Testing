#!/usr/bin/env python3
# 6h_Adaptive_Volatility_Squeeze_Breakout
# Hypothesis: Uses Bollinger Bands width (volatility squeeze) from 1d timeframe to identify low-volatility periods, 
# then breaks out in the direction of 12h trend (EMA50) with volume confirmation. 
# Works in bull/bear by only trading breakouts from squeezes, avoiding whipsaws in high volatility.

name = "6h_Adaptive_Volatility_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility squeeze and trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Bollinger Bands (20, 2) on 1d close
    bb_length = 20
    bb_mult = 2.0
    sma_20 = pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).mean().values
    std_20 = pd.Series(close_1d).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = sma_20 + bb_mult * std_20
    lower_bb = sma_20 - bb_mult * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Squeeze condition: BB width below 20-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma  # True when in low volatility squeeze
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current 6h volume > 1.5x 20-period 6h MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Align 12h trend to 6h
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough history for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(squeeze_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout from squeeze: price breaks above upper BB or below lower BB
            # Only trade in direction of 12h trend with volume confirmation
            if (close[i] > upper_bb_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                squeeze_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lower_bb_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  squeeze_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band (mean reversion) or squeeze ends
            if close[i] < sma_20_aligned[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band or squeeze ends
            if close[i] > sma_20_aligned[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals