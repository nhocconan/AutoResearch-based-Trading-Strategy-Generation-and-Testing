#!/usr/bin/env python3
"""
6h Elder Ray Index with 1d EMA34 Trend and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength relative to trend.
In bull markets (price > 1d EMA34), buy when Bull Power turns positive with volume confirmation.
In bear markets (price < 1d EMA34), sell when Bear Power turns negative with volume confirmation.
Uses 6h timeframe with 1d HTF for trend and power calculation. Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for EMA trend and Elder Ray calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 13-period EMA on 1d close for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(
        span=13, adjust=False, min_periods=13
    ).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components on 1d
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 20-period volume MA for 6h volume confirmation
    vol_ma_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_6h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and Elder Ray
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_6h = vol_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_6h
        
        if position == 0:
            # Look for entry signals
            # Long: bull market (price > 1d EMA34) AND bull power turning positive AND volume confirmation
            long_entry = (curr_close > ema_trend and 
                         bull_power > 0 and 
                         volume_confirm)
            # Short: bear market (price < 1d EMA34) AND bear power turning negative AND volume confirmation
            short_entry = (curr_close < ema_trend and 
                          bear_power < 0 and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below 1d EMA34 OR bull power turns negative
            if curr_close < ema_trend or bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above 1d EMA34 OR bear power turns positive
            if curr_close > ema_trend or bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0