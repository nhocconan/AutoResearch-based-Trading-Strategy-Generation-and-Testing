#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_TrendFilter
Hypothesis: Breakouts above daily R1 or below daily S1 on 12h timeframe with volume confirmation and trend filter (price > EMA34) yield high-probability trades. Targets 15-30 trades/year by requiring strong momentum alignment. Works in bull/bear markets by only taking breakouts in direction of trend. Uses 12h as primary timeframe with 1d HTF for pivot calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate EMA with proper handling of initial values"""
    ema = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period:
        return ema
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(close[:period])
    for i in range(period, len(close)):
        ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = calculate_ema(close_1d, 34)
    
    # Align daily indicators to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Camarilla levels from previous day's OHLC
        # Use prior bar's aligned daily values (previous completed day)
        prev_high = high_1d_aligned[i-1]
        prev_low = low_1d_aligned[i-1]
        prev_close = close_1d_aligned[i-1]
        
        r1, s1 = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA34 for bullish bias, price < EMA34 for bearish bias
        bullish_trend = price > ema34_1d_aligned[i]
        bearish_trend = price < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + bullish trend
            if price > r1 and volume_ok and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + bearish trend
            elif price < s1 and volume_ok and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns bearish
            if price < s1 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns bullish
            if price > r1 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_TrendFilter"
timeframe = "12h"
leverage = 1.0