#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 6h aligned with weekly Camarilla pivot direction (R4/S4) and volume spikes capture strong moves. 
Weekly pivot acts as structural support/resistance: price above weekly R4 = bullish bias, below weekly S4 = bearish bias. 
Volume confirmation ensures breakout conviction. Discrete sizing (0.25) balances return and fee drag. 
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Donchian calculation (need daily high/low for 6h aggregation)
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for weekly Camarilla pivot (R4, S4)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-period Donchian channels on 1d data (then align to 6h)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly Camarilla levels (R4, S4) from prior week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    range_1w = high_1w - low_1w
    camarilla_r4_1w = close_1w + 1.5 * range_1w  # R4 = close + 1.5*range
    camarilla_s4_1w = close_1w - 1.5 * range_1w  # S4 = close - 1.5*range
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to primary timeframe (6h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)  # volume is LTF, but confirm using 1d avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), weekly Camarilla (1), volume avg (20)
    start_idx = max(20, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine weekly trend bias: price vs weekly R4/S4
            bullish_bias = close_val > r4  # Above weekly R4 = bullish
            bearish_bias = close_val < s4  # Below weekly S4 = bearish
            
            if bullish_bias and vol_conf:
                # Long bias: long when price breaks above Donchian high with volume
                if close_val > donchian_high:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif bearish_bias and vol_conf:
                # Short bias: short when price breaks below Donchian low with volume
                if close_val < donchian_low:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss (2.0*ATR) or Donchian low touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < donchian_low:  # Donchian low touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss (2.0*ATR) or Donchian high touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > donchian_high:  # Donchian high touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0