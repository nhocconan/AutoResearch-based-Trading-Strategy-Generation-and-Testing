#!/usr/bin/env python3
"""
1D_1W_Camarilla_R1S1_Breakout_Volume_V1
Hypothesis: Use weekly trend to filter daily R1/S1 breakouts. Long when price breaks above daily R1 with volume > 1.5x average AND weekly close > weekly open (bullish weekly candle). Short when price breaks below daily S1 with volume > 1.5x average AND weekly close < weekly open (bearish weekly candle). Target 10-25 trades/year per symbol to avoid fee drag. Works in bull/bear via weekly trend filter.
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
    
    # Get daily data for R1/S1
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # bullish weekly candle
    weekly_bearish = close_1w < open_1w  # bearish weekly candle
    
    # Align all data to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND weekly bullish
            if close[i] > r1_aligned[i] and vol_confirm[i] and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation AND weekly bearish
            elif close[i] < s1_aligned[i] and vol_confirm[i] and weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or weekly turns bearish
            if close[i] < r1_aligned[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or weekly turns bullish
            if close[i] > s1_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_1W_Camarilla_R1S1_Breakout_Volume_V1"
timeframe = "1d"
leverage = 1.0