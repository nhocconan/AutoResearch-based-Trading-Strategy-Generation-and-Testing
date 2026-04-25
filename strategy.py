#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on 1d timeframe.
Breaks above H3 or below L3 with daily EMA trend alignment and volume spikes capture strong moves.
Uses 12h timeframe for entries with 1d HTF for pivot calculation and trend filter.
Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.
Works in both bull (breakouts with trend) and bear (breakdowns against trend) markets.
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
    
    # Get 1d data for Camarilla pivots and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d (based on previous day's OHLC)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    # We use previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot levels (only valid after we have previous day)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align to 12h timeframe (wait for 1d bar to close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA and to have previous day data
    start_idx = max(20, 1)  # 20 for volume MA, 1 for previous day pivot
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 (resistance), above 1d EMA, volume confirmation
            long_entry = (curr_close > h3_level and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below L3 (support), below 1d EMA, volume confirmation
            short_entry = (curr_close < l3_level and 
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
            # Exit: price falls below L3 (support) OR below 1d EMA
            if curr_close < l3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 (resistance) OR above 1d EMA
            if curr_close > h3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_DailyEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0