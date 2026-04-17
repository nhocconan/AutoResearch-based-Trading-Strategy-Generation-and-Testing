#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Williams %R for mean reversion + volume confirmation + ADX regime filter.
Long when price pulls back to oversold Williams %R (< -80) with volume > 1.5x 24-period average and ADX < 25 (range market).
Short when price rallies to overbought Williams %R (> -20) with volume confirmation and ADX < 25.
Uses Williams %R from 1d for institutional overbought/oversold levels, volume to confirm mean reversion pressure,
and ADX to ensure we're in a ranging market where mean reversion works. Designed to work in both bull (buy pullbacks in uptrend within range) 
and bear (sell rallies in downtrend within range) markets by requiring range regime (ADX < 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h ADX (14-period) for regime filter
    # ADX requires +DI and -DI calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # seed with SMA
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period_adx = 14
    atr = wilders_smooth(tr, period_adx)
    plus_di = 100 * wilders_smooth(plus_dm, period_adx) / (atr + 1e-10)
    minus_di = 100 * wilders_smooth(minus_dm, period_adx) / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smooth(dx, period_adx)
    
    # Calculate 12h volume 24-period average for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align 1d Williams %R and ADX to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(30, 24)  # need enough for Williams %R, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 24-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_24[i]
        
        # Range regime: ADX < 25 (not trending)
        range_regime = adx_aligned[i] < 25
        
        if position == 0:
            # Long: price pulls back to oversold Williams %R (< -80) with volume and range regime
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                range_regime):
                signals[i] = 0.25
                position = 1
            # Short: price rallies to overbought Williams %R (> -20) with volume and range regime
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  range_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or stops oversold
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or stops overbought
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dWilliamsR_MeanReversion_Volume_ADXRange"
timeframe = "12h"
leverage = 1.0