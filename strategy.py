#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R2/S2) breakout with volume confirmation and 1w EMA200 trend filter
- Uses wider Camarilla levels (R2/S2) for higher-probability breakouts with less noise
- Volume confirmation ensures institutional participation
- 1w EMA200 provides strong long-term trend filter to avoid counter-trend trades
- Designed for fewer trades (target: 15-25/year) to minimize fee drag
- Works in bull markets (buying R2 breakouts in uptrend) and bear markets (selling S2 breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA200 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get 4h data for volume average
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla pivot levels (R2, S2) from 1d OHLC
    def calculate_camarilla(high_arr, low_arr, close_arr):
        # Typical price
        pp = (high_arr + low_arr + close_arr) / 3.0
        # Range
        rng = high_arr - low_arr
        # Camarilla levels
        r2 = pp + (rng * 1.1 / 6)
        s2 = pp - (rng * 1.1 / 6)
        return r2, s2
    
    camarilla_r2_1d, camarilla_s2_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate EMA200 on 1w for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (50-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2_1d)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r2_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        ema_trend = ema200_1w_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R2 + volume spike + price > 1w EMA200 (uptrend)
            if price > r2 and vol > 1.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 + volume spike + price < 1w EMA200 (downtrend)
            elif price < s2 and vol > 1.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint between R2 and S2
            mid_point = (r2 + s2) / 2.0
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint between R2 and S2
            mid_point = (r2 + s2) / 2.0
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R2S2_1wEMA200_Volume"
timeframe = "4h"
leverage = 1.0