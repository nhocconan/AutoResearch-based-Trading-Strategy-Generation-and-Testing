#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 in 4h uptrend with volume spike; short when breaks below S1 in 4h downtrend with volume spike.
Uses 1h for entry timing, 4h for trend and levels. Discrete size 0.20 to limit trades (~15-30/year). Works in bull/bear via 4h trend filter.
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
    
    # Load 4h data ONCE before loop for HTF trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate previous 4h bar's high, low, close for Camarilla levels
    # We use the completed 4h bar (so we don't use current forming bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R1 and S1 for previous 4h bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1_4h = close_4h + 1.1 * (high_4h - low_4h) / 12
    camarilla_s1_4h = close_4h - 1.1 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe (already delayed by align_htf_to_ltf for completed bar)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Calculate volume spike filter on 1h: volume > 1.5 * 20-period EMA of volume
    volume_series = pd.Series(volume)
    volume_ema_20 = volume_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ema_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for 4h EMA, 20 for volume EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_r1_4h_aligned[i]) or
            np.isnan(camarilla_s1_4h_aligned[i]) or
            np.isnan(volume_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (fixed_size if position == 1 else -fixed_size)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_4h_aligned[i]
        r1_val = camarilla_r1_4h_aligned[i]
        s1_val = camarilla_s1_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R1, 4h uptrend, volume spike
            long_entry = (close_val > r1_val) and (close_val > ema_50_val) and vol_spike
            # Short: price breaks below S1, 4h downtrend, volume spike
            short_entry = (close_val < s1_val) and (close_val < ema_50_val) and vol_spike
            
            if long_entry:
                signals[i] = fixed_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -fixed_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or price breaks below S1 (reversion to mean)
            if close_val < ema_50_val or close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on trend reversal or price breaks above R1 (reversion to mean)
            if close_val > ema_50_val or close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0