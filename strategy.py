#!/usr/bin/env python3
"""
4h_WaveTrend_Trend_Scalp
Strategy: 4h WaveTrend oscillator with 12h trend filter and volume confirmation.
Long: WT crosses above -60 + 12h uptrend + volume > 1.5x 12-period average
Short: WT crosses below 60 + 12h downtrend + volume > 1.5x 12-period average
Exit: WT crosses back through 0 or trend reversal
Position size: 0.25
Designed to catch momentum swings in trending markets while filtering chop.
Timeframe: 4h
"""

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
    
    # Calculate WaveTrend (WT) oscillator
    # WT1 = EMA(EMA((hlc3 - ema) / (0.015 * mad)), n1)
    # WT2 = SMA(WT1, n2)
    # Where hlc3 = (high + low + close) / 3
    # ema = EMA(hlc3, n1)
    # mad = mean absolute deviation
    
    n1 = 10
    n2 = 21
    
    hlc3 = (high + low + close) / 3.0
    
    # First EMA of hlc3
    ema1 = pd.Series(hlc3).ewm(span=n1, adjust=False).mean().values
    
    # Deviation
    dev = hlc3 - ema1
    
    # Mean absolute deviation
    mad = pd.Series(np.abs(dev)).ewm(span=n1, adjust=False).mean().values
    
    # WT1
    wi = np.where(mad != 0, dev / (0.015 * mad), 0)
    wt1 = pd.Series(wi).ewm(span=n1, adjust=False).mean().values
    wt1 = pd.Series(wt1).ewm(span=n1, adjust=False).mean().values  # Double smoothed
    
    # WT2 = SMA of WT1
    wt2 = pd.Series(wt1).rolling(window=n2, min_periods=n2).mean().values
    
    # Calculate 12h trend (close > open = uptrend, close < open = downtrend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    trend_12h = (df_12h['close'] > df_12h['open']).astype(float).values  # 1 for up, 0 for down
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate 4h volume average (12-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma12_4h = pd.Series(volume_4h).rolling(window=12, min_periods=12).mean().values
    volume_ma12_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma12_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(max(n1*2, n2*2, 12), n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(wt1[i]) or np.isnan(wt2[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(volume_ma12_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma12_4h_aligned[i])
        
        # Trend filter: 12h bullish/bearish
        trend_up = trend_12h_aligned[i] > 0.5  # 12h close > open
        trend_down = trend_12h_aligned[i] < 0.5  # 12h close < open
        
        # WaveTrend signals
        wt1_cross_up = wt1[i-1] < wt2[i-1] and wt1[i] > wt2[i]  # WT1 crosses above WT2
        wt1_cross_down = wt1[i-1] > wt2[i-1] and wt1[i] < wt2[i]  # WT1 crosses below WT2
        wt_cross_zero_up = wt1[i-1] < 0 and wt1[i] >= 0  # WT1 crosses above zero
        wt_cross_zero_down = wt1[i-1] > 0 and wt1[i] <= 0  # WT1 crosses below zero
        
        # Entry signals
        if position == 0:
            # Long: WT1 crosses above WT2 + volume filter + 12h uptrend
            if wt1_cross_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: WT1 crosses below WT2 + volume filter + 12h downtrend
            elif wt1_cross_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WT1 crosses below zero or 12h trend turns down
            if wt_cross_zero_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WT1 crosses above zero or 12h trend turns up
            if wt_cross_zero_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WaveTrend_Trend_Scalp"
timeframe = "4h"
leverage = 1.0