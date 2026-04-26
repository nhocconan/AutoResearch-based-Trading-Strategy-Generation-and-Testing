#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Using 1h timeframe with 4h Camarilla R1/S1 breakouts in direction of 4h EMA50 trend, confirmed by volume spike (>1.5x 24-bar MA). Designed for moderate frequency (target 15-37 trades/year) to avoid fee drag on 1h, works in bull/bear via 4h trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous 4h bar's OHLC for Camarilla levels (R1/S1)
    if len(df_4h) < 2:
        return np.zeros(n)
    
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: volume > 1.5x 24-period average (equivalent to 6h MA on 1h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # Position size
    
    # Warmup: max of calculations
    start_idx = max(24, 1, 50, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close_val > ema_50_val
        bearish_4h = close_val < ema_50_val
        
        # Entry conditions
        long_entry = (close_val > r1_val) and bullish_4h and vol_spike
        short_entry = (close_val < s1_val) and bearish_4h and vol_spike
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r1_val or not bullish_4h):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s1_val or not bearish_4h):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0