#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h strategy using Camarilla R1/S1 breakouts with 4h EMA20 trend filter and volume confirmation. R1/S1 levels provide tighter entries than R3/S3, while 4h EMA20 filters for intermediate-term trend alignment. Volume spike confirms institutional participation. Designed for BTC/ETH robustness via trend filter and session restriction (08-20 UTC). Targets 60-150 total trades over 4 years (15-37/year) with 0.20 position size. Uses discrete levels to minimize fee drag.
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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_20 = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Get 4h data for Camarilla R1/S1 levels (from previous completed 4h bar)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.0833)  # R1 level
    s1 = prev_close - (rng * 1.0833)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Fixed position size to minimize churn
    
    # Warmup: need 4h EMA20 (20), 4h shift(1) for Camarilla, vol avg (20)
    start_idx = max(20 + 1, 1 + 1, 20)
    
    for i in range(start_idx, n):
        # Skip if outside session or any data not ready
        if not in_session[i] or \
           (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_20_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 4h EMA20 alignment and volume confirmation
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA20 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 4h EMA20 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0