#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrendFilter_WeeklyVolume_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and weekly volume confirmation.
- Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Camarilla R3/S3 levels from previous 1d provide institutional support/resistance
- Breakout confirmed by 1d trend alignment (EMA34) and weekly volume spike (>1.5x 20-period MA)
- Designed to capture strong momentum moves while filtering false breakouts in choppy markets
- Works in both bull/bear via trend-following logic with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    # R4 = close + (high - low) * 1.1/2, S4 = close - (high - low) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    r4 = close_1d + range_1d * 1.1 / 2
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly volume spike (>1.5x 20-period MA)
    volume_1w = df_1w['volume'].values
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (vol_ma20_1w * 1.5)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_spike_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        price_above_r3 = close[i] > r3_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        price_above_r4 = close[i] > r4_aligned[i]
        price_below_s4 = close[i] < s4_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Weekly volume confirmation
        vol_spike = volume_spike_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend AND weekly volume spike
            # Use R4 breakout for stronger confirmation when price shows momentum
            if ((price_above_r3 and trend_up and vol_spike) or 
                (price_above_r4 and trend_up)):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend AND weekly volume spike
            elif ((price_below_s3 and trend_down and vol_spike) or 
                  (price_below_s4 and trend_down)):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S3 OR 1d trend turns down
            if price_below_s3 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R3 OR 1d trend turns up
            if price_above_r3 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrendFilter_WeeklyVolume_v1"
timeframe = "6h"
leverage = 1.0