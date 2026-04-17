#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 12h/1d HTF Camarilla R1/S1 breakout + volume confirmation + EMA34 filter.
Long when price breaks above 12h R1 with 1d EMA34 > previous close and volume > 1.5x 20-period 6h volume average.
Short when price breaks below 12h S1 with 1d EMA34 < previous close and volume > 1.5x 20-period 6h volume average.
Uses 12h Camarilla for structure, 1d EMA34 for trend filter, and volume for confirmation.
Designed to capture intraday momentum with HTF structure in both bull and bear markets.
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
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12h Camarilla R1 and S1
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Camarilla levels based on previous day's range
        # R1 = Close + 1.1*(High-Low)/12
        # S1 = Close - 1.1*(High-Low)/12
        rng = high_vals - low_vals
        r1 = close_vals + 1.1 * rng / 12
        s1 = close_vals - 1.1 * rng / 12
        return r1, s1
    
    camarilla_r1_12h, camarilla_s1_12h = camarilla_levels(high_12h, low_12h, close_12h)
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h volume 20-period average
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (6h)
    camarilla_r1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_6h)  # using 12h as reference for alignment
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_12h_aligned[i]) or 
            np.isnan(camarilla_s1_12h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_6h_aligned[i]
        
        # Trend filter: 1d EMA34 direction
        ema_trend_up = ema_34_1d_aligned[i] > close[i-1] if i > 0 else False
        ema_trend_down = ema_34_1d_aligned[i] < close[i-1] if i > 0 else False
        
        if position == 0:
            # Long: price breaks above 12h R1 with uptrend and volume
            if (close[i] > camarilla_r1_12h_aligned[i] and 
                ema_trend_up and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S1 with downtrend and volume
            elif (close[i] < camarilla_s1_12h_aligned[i] and 
                  ema_trend_down and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 12h S1 (opposite side)
            if close[i] < camarilla_s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 12h R1 (opposite side)
            if close[i] > camarilla_r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hCamarilla_R1S1_Volume_1dEMA34Filter"
timeframe = "6h"
leverage = 1.0