#!/usr/bin/env python3
# 6h_1d_engulfing_volume_v1
# Hypothesis: Trade bullish/bearish engulfing patterns on 6h timeframe with volume confirmation and 1d trend filter.
# Engulfing patterns signal potential reversals. Volume confirms conviction. 1d trend filter ensures alignment with higher timeframe momentum.
# Works in both bull and bear markets by filtering trades based on 1d EMA50 trend direction.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_engulfing_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulf = (close[i] > open_price[i]) and (open_price[i] < close[i-1]) and (close[i] > open_price[i-1])
        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulf = (close[i] < open_price[i]) and (open_price[i] > close[i-1]) and (close[i] < open_price[i-1])
        
        if position == 1:  # Long position
            # Exit: bearish engulfing pattern
            if bearish_engulf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish engulfing pattern
            if bullish_engulf:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish engulfing + volume surge + 1d uptrend (price > EMA50)
            if bullish_engulf and vol_surge and (close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bearish engulfing + volume surge + 1d downtrend (price < EMA50)
            elif bearish_engulf and vol_surge and (close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals