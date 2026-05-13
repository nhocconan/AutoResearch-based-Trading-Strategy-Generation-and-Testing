#!/usr/bin/env python3
# Hypothesis: 12h 123 Reversal Pattern with 1w Trend Filter and Volume Confirmation.
# Uses 123 Reversal Pattern (1d) to identify potential reversal points, enters on breakout
# of point 2 in the direction of the 1w trend (EMA50). Volume filter ensures breakout
# has participation. Designed for low trade frequency (~10-20/year) to minimize fee drag.
# 123 pattern works well in ranging markets while trend filter avoids counter-trend trades.

name = "12h_123Reversal_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_123_pattern(high, low, close):
    """Calculate 123 reversal pattern points.
    Point 1: Recent swing high/low
    Point 2: Pullback from point 1
    Point 3: Failed test of point 1 level
    Returns: (point1_high, point2_low, point3_high) for bullish pattern
             (point1_low, point2_high, point3_low) for bearish pattern
    """
    n = len(high)
    point1_high = np.full(n, np.nan)
    point2_low = np.full(n, np.nan)
    point3_high = np.full(n, np.nan)
    point1_low = np.full(n, np.nan)
    point2_high = np.full(n, np.nan)
    point3_low = np.full(n, np.nan)
    
    # Look for swing points over 5-period window
    for i in range(5, n):
        # Bullish 123: High-Low-High
        if (high[i-2] == np.max(high[i-4:i+1]) and 
            low[i] == np.min(low[i-2:i+3]) and 
            high[i] == np.max(high[i-2:i+3])):
            point1_high[i] = high[i-4]  # Swing high before pullback
            point2_low[i] = low[i]      # Pullback low
            point3_high[i] = high[i]    # Failed test of high
            
        # Bearish 123: Low-High-Low
        if (low[i-2] == np.min(low[i-4:i+1]) and 
            high[i] == np.max(high[i-2:i+3]) and 
            low[i] == np.min(low[i-2:i+3])):
            point1_low[i] = low[i-4]    # Swing low before pullback
            point2_high[i] = high[i]    # Pullback high
            point3_low[i] = low[i]      # Failed test of low
    
    return point1_high, point2_low, point3_high, point1_low, point2_high, point3_low

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 123 pattern
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 123 pattern on 1d
    ph1, pl2, ph3, pl1, ph2, pl3 = calculate_123_pattern(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Need 2 extra bars for 123 pattern confirmation
    ph1_aligned = align_htf_to_ltf(prices, df_1d, ph1, additional_delay_bars=2)
    pl2_aligned = align_htf_to_ltf(prices, df_1d, pl2, additional_delay_bars=2)
    ph3_aligned = align_htf_to_ltf(prices, df_1d, ph3, additional_delay_bars=2)
    pl1_aligned = align_htf_to_ltf(prices, df_1d, pl1, additional_delay_bars=2)
    ph2_aligned = align_htf_to_ltf(prices, df_1d, ph2, additional_delay_bars=2)
    pl3_aligned = align_htf_to_ltf(prices, df_1d, pl3, additional_delay_bars=2)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 30-period average
    volume_series = pd.Series(volume)
    vol_ma30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > vol_ma30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma30[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above point 3 of bullish 123 with uptrend and volume
            if (not np.isnan(ph3_aligned[i]) and 
                close[i] > ph3_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below point 3 of bearish 123 with downtrend and volume
            elif (not np.isnan(pl3_aligned[i]) and 
                  close[i] < pl3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below point 2 of bullish 123 (pattern failure)
            if not np.isnan(pl2_aligned[i]) and close[i] < pl2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above point 2 of bearish 123 (pattern failure)
            if not np.isnan(ph2_aligned[i]) and close[i] > ph2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals