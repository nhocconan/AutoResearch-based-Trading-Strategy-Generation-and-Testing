#!/usr/bin/env python3
"""
4h_Engulfing_Engulfing_VolumeFilter_V1
Strategy: 4h bullish/bearish engulfing candle with volume confirmation.
Long: Bullish engulfing pattern + volume > 1.5x 20-period average
Short: Bearish engulfing pattern + volume > 1.5x 20-period average
Exit: Opposite engulfing pattern or time-based (max 10 bars)
Position size: 0.25
Uses candlestick patterns for reversal signals, volume for confirmation.
Designed to capture reversals in both trending and ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    for i in range(50, n):  # warmup for EMA
        # Skip if EMA not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Engulfing patterns
        bullish_engulfing = (close[i] > open_price[i-1]) and (open_price[i] < close[i-1])
        bearish_engulfing = (open_price[i] > close[i-1]) and (close[i] < open_price[i-1])
        
        if position == 0:
            # Long: Bullish engulfing + above EMA34 + volume filter
            if bullish_engulfing and (close[i] > ema_34_1d_aligned[i]) and volume_filter:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 0
            # Short: Bearish engulfing + below EMA34 + volume filter
            elif bearish_engulfing and (close[i] < ema_34_1d_aligned[i]) and volume_filter:
                signals[i] = -0.25
                position = -1
                bars_in_trade = 0
        
        elif position == 1:
            bars_in_trade += 1
            # Exit conditions: bearish engulfing OR max 10 bars
            if bearish_engulfing or bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            bars_in_trade += 1
            # Exit conditions: bullish engulfing OR max 10 bars
            if bullish_engulfing or bars_in_trade >= 10:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Engulfing_Engulfing_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0