#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_1dTrendFilter
Hypothesis: Williams Vix Fix identifies extreme fear/greed on 6h; mean reversion when WVF > 0.8 (fear) or < 0.2 (greed) with 1d trend filter (price > 1d EMA50 for long WVF mean reversion, price < 1d EMA50 for short WVF mean reversion). Volume confirmation (>1.5x 20-bar mean) ensures conviction. Designed for 12-25 trades/year per symbol, effective in ranging markets (mean reversion) and trending markets (filter avoids counter-trend trades). Works in both bull (buy fear dips in uptrend) and bear (sell greed spikes in downtrend).
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Vix Fix: measures fear/greed, similar to VIX but for crypto
    # WVF = ((Highest Close in period - Low) / (Highest Close in period - Highest Close in period)) * 100
    # Normalized to 0-1, where >0.8 = extreme fear, <0.2 = extreme greed
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    wvf = (highest_close - low) / (highest_close - highest_close)  # This will have division by zero
    # Fix: avoid division by zero when highest_close == highest_close (always true)
    # Correct WVF formula: (Highest Close - Low) / (Highest Close - Lowest Close in period)
    lowest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    denominator = highest_close - lowest_close
    denominator = np.where(denominator == 0, 1, denominator)  # replace zeros with 1
    wvf = (highest_close - low) / denominator
    wvf = np.nan_to_num(wvf, nan=0.0)  # replace nan with 0
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for WVF and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(wvf[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: extreme fear (WVF > 0.8) in uptrend (price > 1d EMA50) with volume confirmation
            # Short: extreme greed (WVF < 0.2) in downtrend (price < 1d EMA50) with volume confirmation
            long_signal = (wvf[i] > 0.8) and (close[i] > ema_50_aligned[i]) and vol_confirm[i]
            short_signal = (wvf[i] < 0.2) and (close[i] < ema_50_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when fear subsides (WVF < 0.5) or trend reverses
            exit_signal = (wvf[i] < 0.5) or (close[i] < ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when greed subsides (WVF > 0.5) or trend reverses
            exit_signal = (wvf[i] > 0.5) or (close[i] > ema_50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_1dTrendFilter"
timeframe = "6h"
leverage = 1.0