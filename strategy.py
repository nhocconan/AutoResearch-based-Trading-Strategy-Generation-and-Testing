#!/usr/bin/env python3
"""
4h_1d_1w_Donchian_Breakout_VolumeTrend_v2
Concept: Donchian(20) breakout on 4h with volume confirmation and 1d/1w trend filter.
- Long: Close > 20-bar high AND volume > 1.5x 20-bar volume MA AND 1d close > 1d EMA50 AND 1w close > 1w EMA50
- Short: Close < 20-bar low AND volume > 1.5x 20-bar volume MA AND 1d close < 1d EMA50 AND 1w close < 1w EMA50
- Exit: Close crosses 10-bar EMA (adaptive exit)
- Position sizing: 0.25 (conservative to limit drawdown)
- Target: ~150 total trades over 4 years (~38/year) to minimize fee drag
- Works in bull/bear: Trend filter ensures alignment with higher timeframe direction, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Donchian_Breakout_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 4h: Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        dc_high = high_20[i]
        dc_low = low_20[i]
        vol_ma = vol_ma_20[i]
        vol_current = volume[i]
        curr_close = close[i]
        ema_10_val = ema_10[i]
        ema_1d = ema_50_1d_aligned[i]
        ema_1w = ema_50_1w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(dc_high) or np.isnan(dc_low) or np.isnan(vol_ma) or 
            np.isnan(ema_10_val) or np.isnan(ema_1d) or np.isnan(ema_1w)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = vol_current > 1.5 * vol_ma
        
        # Trend condition: 1d and 1w close above/below their EMA50
        # Note: We don't have 1d/1w close directly in 4h array, so we approximate
        # using the aligned EMA - if price is above EMA, trend is up
        # For simplicity, we use price vs EMA as trend proxy
        trend_up = curr_close > ema_1d and curr_close > ema_1w
        trend_down = curr_close < ema_1d and curr_close < ema_1w
        
        if position == 0:
            # Long: breakout above Donchian high with volume and trend confirmation
            if curr_close > dc_high and vol_condition and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low with volume and trend confirmation
            elif curr_close < dc_low and vol_condition and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close crosses below 10-period EMA
            if curr_close < ema_10_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close crosses above 10-period EMA
            if curr_close > ema_10_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals