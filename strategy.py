#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout with 12h EMA50 Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance.
Breakouts above H3 or below L3 with 12h EMA trend alignment and volume spikes
capture strong moves. Uses 4h timeframe with 12h HTF for trend filter.
Targets 75-200 trades over 4 years (19-50/year) to avoid fee drag.
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
    
    # Get 12h data for Camarilla pivots and EMA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    # Camarilla: based on previous period's high, low, close
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Avoid look-ahead: use only completed previous period
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels (H3/L3)
    H3 = typical_price + range_hl * 1.1 / 4.0
    L3 = typical_price - range_hl * 1.1 / 4.0
    
    # Calculate 50-period EMA on 12h close
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(df_12h['volume'].values[i-19:i+1])
    
    # Align to 4h
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 20-period volume MA for 4h volume spike
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for volume MA and pivots
    start_idx = max(20, 1)  # 20 for volume MA, 1 for pivot shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        vol_ma_12h = vol_ma_20_12h_aligned[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 2.0 * 20-period 4h average
        volume_confirm = curr_volume > 2.0 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3, above 12h EMA, volume confirmation
            long_entry = (curr_close > H3 and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below L3, below 12h EMA, volume confirmation
            short_entry = (curr_close < L3 and 
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
            # Exit: price falls below L3 OR below 12h EMA
            if curr_close < L3 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above H3 OR above 12h EMA
            if curr_close > H3 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0