#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS
# Hypothesis: Use Camarilla pivot R1/S1 levels from 1d as entry triggers with 1d EMA trend filter and volume confirmation.
# In bull markets: buy breakouts above R1 when 1d EMA34 is rising and volume spikes.
# In bear markets: sell breakdowns below S1 when 1d EMA34 is falling and volume spikes.
# The Camarilla levels provide precise support/resistance, EMA34 filters trend direction,
# and volume spikes confirm institutional participation. This combination reduces false breakouts.
# Target: 20-30 trades/year with low turnover to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Volume spike detection: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 1d
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    
    # Previous day's typical price for pivot calculation
    prev_typical = np.roll(typical_price_vals, 1)
    prev_typical[0] = np.nan  # First value has no previous day
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_width = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_width
    s1 = close_1d - camarilla_width
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 1d EMA34 rising
            if close[i] > r1_aligned[i] and volume_spike[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and 1d EMA34 falling
            elif close[i] < s1_aligned[i] and volume_spike[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or volume drops
            if close[i] < s1_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or volume drops
            if close[i] > r1_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals