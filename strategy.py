#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction and volume average for spike confirmation.
- Camarilla levels from 1d: H3 (resistance 3) and L3 (support 3) as breakout levels.
- Trend filter: 12h EMA50 slope > 0 for longs, < 0 for shorts (avoid counter-trend trades).
- Volume confirmation: current volume > 2.0 * 20-period average volume on 12h timeframe.
- Entry: Long when price > H3 AND 12h EMA50 rising AND volume spike.
         Short when price < L3 AND 12h EMA50 falling AND volume spike.
- Exit: Opposite Camarilla level (price < H3 for long exit, price > L3 for short exit) or EMA trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 12h trend, avoiding whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for calculation
        return np.zeros(n)
    
    # Camarilla calculation uses previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + (prev_range * 1.1 / 4)
    camarilla_l3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h EMA50 slope (trend direction)
    ema50_slope = np.diff(ema50_12h_aligned, prepend=ema50_12h_aligned[0])
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_slope[i]) or
            np.isnan(vol_ma_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: EMA50 slope > 0 for uptrend, < 0 for downtrend
        uptrend = ema50_slope[i] > 0
        downtrend = ema50_slope[i] < 0
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_12h_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price < H3 OR trend turns down
            if position == 1:
                if curr_close < camarilla_h3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3 OR trend turns up
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > H3 AND uptrend AND volume confirmation
            long_condition = (curr_close > camarilla_h3_aligned[i] and 
                            uptrend and
                            volume_confirm)
            
            # Short: price < L3 AND downtrend AND volume confirmation
            short_condition = (curr_close < camarilla_l3_aligned[i] and 
                             downtrend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_CamarillaH3L3_Breakout_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0