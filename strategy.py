#!/usr/bin/env python3
# 6h_weekly_pivot_breakout_v1
# Hypothesis: 6h strategy using weekly Camarilla pivot levels for structure, with breakout confirmation on 6h closes. Long when price breaks above weekly R4 with volume > 1.5x 20-bar average; short when price breaks below weekly S4 with volume confirmation. Uses 1d EMA(50) for trend alignment to avoid counter-trend trades. Designed for low frequency (target: 12-37 trades/year) to minimize fee drag on 6s timeframe. Works in bull/bear: breakouts capture momentum, volume confirms conviction, HTF EMA filters weak breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Multi-timeframe: weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for weekly
    camarilla_h4 = close_1w + 1.1 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.1 * (high_1w - low_1w) / 2
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    camarilla_h2 = close_1w + 1.1 * (high_1w - low_1w) / 6
    camarilla_l2 = close_1w - 1.1 * (high_1w - low_1w) / 6
    camarilla_h1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_l1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    camarilla_h4 = camarilla_h4  # R4
    camarilla_l4 = camarilla_l4  # S4
    camarilla_h3 = camarilla_h3  # R3
    camarilla_l3 = camarilla_l3  # S3
    
    # Align weekly levels to 6h timeframe (using previous week's levels for forward-looking safety)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # HTF trend filter: price above/below 1d EMA(50)
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly R3 (profit taking or reversal)
            if close[i] < camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly S3 (profit taking or reversal)
            if close[i] > camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume and HTF confirmation
            bullish_breakout = (close[i] > camarilla_h4_aligned[i]) and volume_confirmed and htf_uptrend
            bearish_breakout = (close[i] < camarilla_l4_aligned[i]) and volume_confirmed and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals