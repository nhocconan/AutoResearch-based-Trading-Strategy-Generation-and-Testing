#!/usr/bin/env python3
# 4h_Bollinger_Squeeze_Breakout
# Hypothesis: Bollinger Bands squeeze (low volatility) precedes breakouts in both bull and bear markets.
# Uses 4h timeframe with 1d Bollinger Bands for volatility regime detection, volume confirmation
# for breakout strength, and ATR-based stoploss. Bollinger Band width < 50th percentile indicates
# low volatility squeeze; breakout occurs when price closes outside Bollinger Bands with volume
# > 1.5x average. Works in ranging markets (captures breakouts from consolidation) and trending
# markets (rides the breakout). Targets 20-40 trades/year to minimize fee drag.

name = "4h_Bollinger_Squeeze_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20, 2) on daily timeframe
    bb_period = 20
    bb_std = 2
    
    # Calculate rolling mean and std
    ma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper_bb = ma + (bb_std * std)
    lower_bb = ma - (bb_std * std)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (50-period lookback)
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(50, len(bb_width)):
        window = bb_width[i-50:i]
        if len(window) > 0 and not np.all(np.isnan(window)):
            bb_width_percentile[i] = np.percentile(window, 50)  # 50th percentile (median)
        else:
            bb_width_percentile[i] = bb_width[i] if not np.isnan(bb_width[i]) else 0
    
    # Squeeze condition: BB width < 50th percentile (low volatility)
    squeeze = bb_width < bb_width_percentile
    
    # Align Bollinger Bands and squeeze to 4h timeframe
    upper_bb_4h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_4h = align_htf_to_ltf(prices, df_1d, lower_bb)
    squeeze_4h = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))
    
    # Volume confirmation: volume > 1.5x 24-period average (24 * 4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_bb_4h[i]) or np.isnan(lower_bb_4h[i]) or 
            np.isnan(squeeze_4h[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout from squeeze with volume confirmation
            if squeeze_4h[i] > 0.5:  # Currently in squeeze
                if close[i] > upper_bb_4h[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_bb_4h[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: price returns to middle Bollinger Band (mean reversion) or opposite band touch
            middle_bb = (upper_bb_4h[i] + lower_bb_4h[i]) / 2
            if close[i] < middle_bb or close[i] > upper_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to middle Bollinger Band or opposite band touch
            middle_bb = (upper_bb_4h[i] + lower_bb_4h[i]) / 2
            if close[i] > middle_bb or close[i] < lower_bb_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals