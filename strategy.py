#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA34 trend filter, volume confirmation (>1.5x 20-bar MA), and session filter (08-20 UTC). Uses higher timeframe (4h) for trend direction and 1h for precise entry timing. Designed to work in both bull and bear markets by following 4h trend while using Camarilla structure for breakout entries. Volume spike reduces whipsaws, session filter avoids low-liquidity periods. Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH.
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
    open_time = prices['open_time'].values
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Previous day's OHLC for Camarilla levels (using 1d for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (primary breakout levels)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)  # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Position size (20% of capital)
    
    # Warmup: max of calculations (20 for vol, 34 for 4h EMA, 1 for camarilla)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            not in_session[i]):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 4h trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_4h = close_val > ema_34_val
        bearish_4h = close_val < ema_34_val
        
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
            # Long - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point or not bullish_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point or not bearish_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0