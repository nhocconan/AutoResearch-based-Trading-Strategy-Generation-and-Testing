#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter, volume confirmation (>1.8x 20-bar MA), and session filter (08-20 UTC). Uses 4h trend for signal direction, 1h for precise entry timing. Designed to work in both bull and bear markets by following 4h trend while using Camarilla structure for breakout entries. Volume spike reduces whipsaws, session filter avoids low-liquidity periods. Target: 15-35 trades/year on BTC/ETH.
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
    
    # Load 4h data ONCE before loop for HTF filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Calculate Camarilla levels: R1, S1 (standard breakout levels)
    rng = high_4h - low_4h
    camarilla_r1 = close_4h_vals + (rng * 1.1 / 2)   # R1 level
    camarilla_s1 = close_4h_vals - (rng * 1.1 / 2)   # S1 level
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: volume > 1.8x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Position size (20% of capital)
    
    # Warmup: max of calculations (20 for vol, 50 for 4h EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        if not in_session[i]:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close_val > ema_50_val
        bearish_4h = close_val < ema_50_val
        
        # Entry conditions: breakout of Camarilla R1/S1 in trend direction with volume spike
        long_entry = (close_val > camarilla_r1_val) and bullish_4h and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_4h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point or not bullish_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to midpoint or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point or not bearish_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0