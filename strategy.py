#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume spike confirmation.
Designed for low trade frequency (target 15-30/year) on 1d timeframe to minimize fee drag while capturing major swings.
Weekly EMA34 provides strong trend filter that adapts to bull/bear regimes. Volume spike confirms institutional interest.
Works on BTC and ETH as primary targets. Uses discrete position sizing (0.25) to balance return and drawdown.
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
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate average volume (20-period) on daily for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous daily bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly EMA(34), volume MA(20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_val = volume[i]
        trend_1w_up = close_val > ema_34_1w_aligned[i]   # Weekly uptrend
        trend_1w_down = close_val < ema_34_1w_aligned[i]  # Weekly downtrend
        volume_spike = vol_val > 2.0 * vol_ma_20[i]       # Volume > 2x average
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND weekly trend up AND volume spike
            long_signal = (close_val > camarilla_r1_aligned[i]) and trend_1w_up and volume_spike
            
            # Short: price breaks below Camarilla S1 AND weekly trend down AND volume spike
            short_signal = (close_val < camarilla_s1_aligned[i]) and trend_1w_down and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly trend flips down
            if not trend_1w_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend flips up
            if not trend_1w_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0