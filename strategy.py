#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_1dTrend_VolumeSpike_v3
Hypothesis: 12h Camarilla H3/L3 breakouts with daily EMA trend filter and volume spike capture institutional breakout moves in both bull and bear markets.
Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
Uses volume spike (1.5x 20-period average) to filter false breakouts and daily EMA50 for trend alignment.
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
    
    # Get daily data for Camarilla calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    rang = prev_high - prev_low
    H3 = prev_close + rang * 1.1 / 4
    L3 = prev_close - rang * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get daily EMA for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above H3 with daily uptrend and volume spike
            if price > h3 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with daily downtrend and volume spike
            elif price < l3 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to L3 or breaks below daily EMA
            if price < l3 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to H3 or breaks above daily EMA
            if price > h3 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_1dTrend_VolumeSpike_v3"
timeframe = "12h"
leverage = 1.0