#!/usr/bin/env python3
"""
6h Bollinger Squeeze + Volume Spike + 1d Trend Filter
Hypothesis: Bollinger Band squeeze indicates low volatility and impending breakout.
Combine with volume spike (institutional participation) and 1d trend filter (close vs SMA50)
to capture explosive moves in both bull and bear markets. Low frequency due to strict
multi-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands, returns (upper, middle, lower)"""
    sma = np.zeros_like(close)
    bb_up = np.zeros_like(close)
    bb_dn = np.zeros_like(close)
    
    for i in range(len(close)):
        if i < period - 1:
            sma[i] = np.nan
            bb_up[i] = np.nan
            bb_dn[i] = np.nan
        else:
            sma[i] = np.mean(close[i-period+1:i+1])
            std = np.std(close[i-period+1:i+1])
            bb_up[i] = sma[i] + std_dev * std
            bb_dn[i] = sma[i] - std_dev * std
    return bb_up, sma, bb_dn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate SMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    sma_50 = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 49:
            sma_50[i] = np.nan
        else:
            sma_50[i] = np.mean(close_1d[i-49:i+1])
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # Bollinger Bands (20,2) on 6h
    bb_up, bb_mid, bb_dn = bollinger_bands(close, period=20, std_dev=2.0)
    bb_width = (bb_up - bb_dn) / bb_mid  # Normalized width
    
    # Bollinger Squeeze: width < 20th percentile of last 50 bars
    bb_width_percentile = np.zeros_like(bb_width)
    for i in range(len(bb_width)):
        if i < 50:
            bb_width_percentile[i] = np.nan
        else:
            window = bb_width[i-49:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bb_width_percentile[i] = (np.sum(valid <= bb_width[i]) / len(valid)) * 100
            else:
                bb_width_percentile[i] = np.nan
    squeeze = bb_width_percentile < 20  # Squeeze when width in bottom 20%
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(sma_50_aligned[i]) or np.isnan(bb_width[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: close vs 1d SMA50
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        if position == 0:
            # Enter long: squeeze + volume spike + uptrend
            if squeeze[i] and vol_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + volume spike + downtrend
            elif squeeze[i] and vol_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: squeeze ends or trend reverses
            if not squeeze[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: squeeze ends or trend reverses
            if not squeeze[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_VolumeSpike_1dTrendFilter"
timeframe = "6h"
leverage = 1.0