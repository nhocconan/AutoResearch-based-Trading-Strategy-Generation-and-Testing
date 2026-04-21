#!/usr/bin/env python3
"""
12h_1d_Engulfing_Pattern_Reversal
Hypothesis: Use 12h timeframe with daily bullish/bearish engulfing patterns for reversal signals.
Long when bullish engulfing forms at support (below 200 EMA) with volume confirmation.
Short when bearish engulfing forms at resistance (above 200 EMA) with volume confirmation.
Exit on opposite engulfing signal or when price crosses 200 EMA.
Works in bull markets by buying dips and in bear markets by selling rallies.
Engulfing patterns signal strong reversals, reducing false signals.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 200 EMA on close prices
    close = prices['close'].values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 1d data for engulfing patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate bullish and bearish engulfing patterns on daily data
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    bullish_engulfing = np.zeros(len(close_1d), dtype=bool)
    bearish_engulfing = np.zeros(len(close_1d), dtype=bool)
    
    for i in range(1, len(close_1d)):
        # Bullish engulfing: current green candle engulfs previous red candle
        if (close_1d[i] > open_1d[i] and  # current candle is green
            open_1d[i-1] > close_1d[i-1] and  # previous candle is red
            close_1d[i] >= open_1d[i-1] and  # current close >= previous open
            open_1d[i] <= close_1d[i-1]):   # current open <= previous close
            bullish_engulfing[i] = True
        
        # Bearish engulfing: current red candle engulfs previous green candle
        if (close_1d[i] < open_1d[i] and  # current candle is red
            open_1d[i-1] < close_1d[i-1] and  # previous candle is green
            close_1d[i] <= open_1d[i-1] and  # current close <= previous open
            open_1d[i] >= close_1d[i-1]):   # current open >= previous close
            bearish_engulfing[i] = True
    
    # Shift patterns to align with next day (patterns known after candle close)
    bullish_engulfing = np.roll(bullish_engulfing, 1)
    bearish_engulfing = np.roll(bearish_engulfing, 1)
    bullish_engulfing[0] = False
    bearish_engulfing[0] = False
    
    # Align to 12h timeframe
    bullish_engulfing_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulfing.astype(float))
    bearish_engulfing_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulfing.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_200[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: bullish engulfing + price below 200 EMA + volume
            if bullish_engulfing_aligned[i] > 0.5 and price < ema_200[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish engulfing + price above 200 EMA + volume
            elif bearish_engulfing_aligned[i] > 0.5 and price > ema_200[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish engulfing or price crosses above 200 EMA
            if bearish_engulfing_aligned[i] > 0.5 or price > ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish engulfing or price crosses below 200 EMA
            if bullish_engulfing_aligned[i] > 0.5 or price < ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Engulfing_Pattern_Reversal"
timeframe = "12h"
leverage = 1.0