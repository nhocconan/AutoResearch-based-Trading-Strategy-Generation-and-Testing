#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 6h Camarilla pivot breakout at R3/S3 levels with 1w trend filter (price >/< 1w EMA34) and volume confirmation (>1.8x 20-bar avg). 
Enters long on break above R3 in 1w uptrend with volume spike, short on break below S3 in 1w downtrend with volume spike. 
Exits on opposite break (below R3 for longs, above S3 for shorts) or trend reversal. 
Uses weekly timeframe for structural trend to avoid whipsaws in ranging markets, targeting 20-40 trades/year.
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
    
    # 1d data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for today using previous day's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = (prev_day_H + prev_day_L + prev_day_C)/3 (typical price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    rng_1d = high_1d - low_1d
    r3_1d = typical_price_1d + (rng_1d * 1.1 / 2.0)
    s3_1d = typical_price_1d - (rng_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (using previous day's levels for today)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 and Camarilla calculation
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 in 1w uptrend with volume confirmation
            long_setup = (close[i] > r3_1d_aligned[i]) and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below S3 in 1w downtrend with volume confirmation
            short_setup = (close[i] < s3_1d_aligned[i]) and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below R3 (failed breakout) OR trend turns down
            if (close[i] < r3_1d_aligned[i]) or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above S3 (failed breakdown) OR trend turns up
            if (close[i] > s3_1d_aligned[i]) or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0