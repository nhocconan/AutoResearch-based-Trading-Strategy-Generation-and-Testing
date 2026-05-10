#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Long when price breaks above Camarilla R3 (strong resistance) with volume > 1.5x average and weekly close above weekly open (bullish weekly candle).
# Short when price breaks below Camarilla S3 (strong support) with volume > 1.5x average and weekly close below weekly open (bearish weekly candle).
# Exit when price crosses the opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses Camarilla pivot levels from daily timeframe for structure, weekly trend filter for direction, and volume confirmation to avoid false breakouts.
# Designed for 12h timeframe to target 12-37 trades/year, avoiding fee drag while capturing significant moves in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily Camarilla levels (using prior day's OHLC)
    close_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels for each day using prior day's data
    high_1d = close_1d['high'].values
    low_1d = close_1d['low'].values
    close_1d_prev = close_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 6
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 6
    
    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, close_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, close_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, close_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, close_1d, S1)
    
    # Weekly trend: bullish if weekly close > weekly open
    close_1w = get_htf_data(prices, '1w')
    weekly_open = close_1w['open'].values
    weekly_close = close_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, close_1w, weekly_bullish.astype(float))
    
    # Volume average (24 periods = 12 days of 12h data)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.nanmean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and bullish weekly trend
            if close[i] > R3_aligned[i] and volume[i] > 1.5 * vol_ma[i] and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation and bearish weekly trend
            elif close[i] < S3_aligned[i] and volume[i] > 1.5 * vol_ma[i] and weekly_bullish_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below S1 (support) or weekly trend turns bearish
            if close[i] < S1_aligned[i] or weekly_bullish_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above R1 (resistance) or weekly trend turns bullish
            if close[i] > R1_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals