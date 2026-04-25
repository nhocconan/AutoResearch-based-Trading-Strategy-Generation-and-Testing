#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3, L3) from weekly data act as strong support/resistance.
Breakouts above H3 or below L3 with weekly EMA trend alignment and volume spikes capture
strong moves. Uses 12h timeframe with 1w HTF for trend and pivot calculation. Targets 50-150
trades over 4 years (12-37/year) to avoid fee drag. Works in both bull and bear markets by
trading breakouts in the direction of the higher timeframe trend.
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
    
    # Get 1w data for Camarilla pivots and EMA (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for weekly data
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    # Camarilla width = (H - L) * 1.1 / 8
    width = (df_1w['high'].values - df_1w['low'].values) * 1.1 / 8.0
    # H3 = typical_price + width * 4
    # L3 = typical_price - width * 4
    H3 = typical_price + width * 4.0
    L3 = typical_price - width * 4.0
    
    # Camarilla pivots need 1 extra bar for confirmation (weekly close)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3, additional_delay_bars=1)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3, additional_delay_bars=1)
    
    # Calculate 34-period EMA on 1w close (only needs completed 1w candle)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 (resistance), above weekly EMA, volume confirmation
            long_entry = (curr_close > H3_level and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below L3 (support), below weekly EMA, volume confirmation
            short_entry = (curr_close < L3_level and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below L3 (support) OR below weekly EMA
            if curr_close < L3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (resistance) OR above weekly EMA
            if curr_close > H3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0