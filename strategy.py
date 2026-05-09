#!/usr/bin/env python3
# 1h_MeanReversion_LB_4hTrend_Volume
# Hypothesis: 1h mean reversion using 20-period Bollinger Bands with 4h trend filter.
# Long when price touches lower Bollinger Band and 4h trend is up (close > 4h EMA20).
# Short when price touches upper Bollinger Band and 4h trend is down (close < 4h EMA20).
# Bollinger Band width < 50th percentile indicates ranging market for mean reversion.
# Uses 4h EMA20 for trend filter to avoid counter-trend trades in strong trends.
# Volume confirmation: current volume > 1.5x 20-period average to filter low-volume noise.
# Designed for low trade frequency (15-30/year) to minimize fee drag on 1h timeframe.
# Works in both bull and bear markets by fading extremes only when 4h trend supports reversion.

name = "1h_MeanReversion_LB_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 20:
        ema20_4h[19] = np.mean(close_4h[0:20])
        for i in range(20, len(close_4h)):
            ema20_4h[i] = (close_4h[i] * 2 + ema20_4h[i-1] * 18) / 20
    
    # Align 4h EMA20 to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    sma = np.full_like(close, np.nan)
    if len(close) >= bb_period:
        sma[bb_period-1] = np.mean(close[0:bb_period])
        for i in range(bb_period, len(close)):
            sma[i] = (sma[i-1] * (bb_period-1) + close[i]) / bb_period
    
    variance = np.full_like(close, np.nan)
    if len(close) >= bb_period:
        for i in range(bb_period-1, len(close)):
            if i == bb_period-1:
                variance[i] = np.mean((close[0:bb_period] - sma[i]) ** 2)
            else:
                variance[i] = (variance[i-1] * (bb_period-1) + (close[i] - sma[i]) ** 2) / bb_period
    
    std_dev = np.sqrt(variance)
    upper_band = sma + bb_std * std_dev
    lower_band = sma - bb_std * std_dev
    
    # Bollinger Band Width for regime filter
    bb_width = (upper_band - lower_band) / sma
    
    # Percentile rank of BB width (50-period lookback)
    bb_width_rank = np.full_like(bb_width, np.nan)
    lookback = 50
    for i in range(lookback-1, len(bb_width)):
        if not np.isnan(bb_width[i-lookback+1:i+1]).any():
            sorted_width = np.sort(bb_width[i-lookback+1:i+1])
            rank = np.searchsorted(sorted_width, bb_width[i]) / lookback * 100
            bb_width_rank[i] = rank
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20, lookback-1, 20)  # Need BB, 4h EMA, BB width rank, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std_dev[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(bb_width_rank[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema20_4h_aligned[i]
        
        # Mean reversion conditions: price at band + ranging market (BB width < 50th percentile)
        at_lower_band = close[i] <= lower_band[i]
        at_upper_band = close[i] >= upper_band[i]
        ranging_market = bb_width_rank[i] < 50
        
        if position == 0:
            # Enter long: price at lower BB + ranging market + 4h trend up + volume
            if at_lower_band and ranging_market and trend_up and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: price at upper BB + ranging market + 4h trend down + volume
            elif at_upper_band and ranging_market and not trend_up and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle (SMA) or 4h trend turns down
            if close[i] >= sma[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to middle (SMA) or 4h trend turns up
            if close[i] <= sma[i] or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals