#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, Camarilla R1/S1 level breakouts with 1w EMA34 trend filter and volume confirmation (>2.0x 20-bar avg) capture institutional moves in both bull and bear markets. Uses weekly trend to avoid counter-trend trades and volume spike to confirm institutional participation. Targets 15-25 trades/year to minimize fee drag while maintaining edge via trend and volume filters.
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
    
    # Get 1d data for Camarilla levels and daily volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(30, 40, 20)  # 1d lookback, 1w lookback, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ma_20_1d_val = vol_ma_20_1d_aligned[i]
        vol_1d_val = volume_1d[i]  # Use current 1d volume (already aligned via index)
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Calculate Camarilla levels for today using previous day's OHLC
        # Need previous day's data (i-1 in 1d array corresponds to current day)
        prev_idx = i - 1
        if prev_idx < 0:
            # Hold position or flat on first bar
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        if np.isnan(close_1d[prev_idx]) or np.isnan(high_1d[prev_idx]) or np.isnan(low_1d[prev_idx]):
            # Hold position or flat if data not ready
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Previous day's OHLC for Camarilla calculation
        prev_close = close_1d[prev_idx]
        prev_high = high_1d[prev_idx]
        prev_low = low_1d[prev_idx]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Hold position or flat if invalid range
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        camarilla_r1 = prev_close + (range_val * 1.1 / 12)
        camarilla_s1 = prev_close - (range_val * 1.1 / 12)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirmed = vol_1d_val > 2.0 * vol_ma_20_1d_val
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with uptrend (close > EMA34) and volume confirmation
            long_signal = (close_val > camarilla_r1) and (close_val > ema_34_1w_val) and volume_confirmed
            # Short: price breaks below Camarilla S1 with downtrend (close < EMA34) and volume confirmation
            short_signal = (close_val < camarilla_s1) and (close_val < ema_34_1w_val) and volume_confirmed
            
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
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price closes below Camarilla S1 (mean reversion)
            if close_val < camarilla_s1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price closes above Camarilla R1 (mean reversion)
            if close_val > camarilla_r1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0