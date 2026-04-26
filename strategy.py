#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike (>2.0x 20-bar avg) capture high-probability institutional breakouts. Using 4h for trend direction reduces noise and whipsaws, while 1d volume confirmation ensures institutional participation. 1h timeframe allows precise entry timing. Session filter (08-20 UTC) reduces off-hours noise. Target position size 0.20 to balance risk and return. Designed to work in both bull and bear markets via trend filter adaptation.
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
    
    # Get 4h data for HTF trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range * 1.0 / 4  # R1 level
    s1 = prev_close - 1.1 * camarilla_range * 1.0 / 4  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d volume average (20-period) for volume spike filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1h volume > 2.0x 20-period 1d average
        # Note: Comparing 1h volume to 1d average volume requires scaling
        # Approximate: 1d volume ~ 6x 4h volume ~ 24x 1h volume (6*4=24)
        volume_spike = vol_val > 2.0 * vol_ma_1d_val / 24.0
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_50_val) and volume_spike
            # Short: price breaks below S1 with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_50_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0