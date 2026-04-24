#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA trend filter (EMA34) and 1d for ATR-based volume spike.
- Camarilla levels (H3/L3): Calculated from prior 1d OHLC: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6.
- Regime: 12h EMA34 slope > 0 = uptrend (favor longs), < 0 = downtrend (favor shorts).
- Entry: Long when price > H3 AND 12h EMA34 up AND volume > 2.0 * 20-period 1d average volume.
         Short when price < L3 AND 12h EMA34 down AND volume > 2.0 * 20-period 1d average volume.
- Exit: Opposite Camarilla break (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear by only trading breaks in direction of 12h trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h EMA34 slope (trend direction: >0 = up, <0 = down)
    # Use 3-bar slope to reduce noise: (current - 3 bars ago) / 3
    ema34_slope = np.zeros_like(ema34_12h_aligned)
    ema34_slope[3:] = (ema34_12h_aligned[3:] - ema34_12h_aligned[:-3]) / 3
    # For first 3 bars, set to 0 (no slope)
    ema34_slope[:3] = 0
    
    # Calculate 1d ATR for volume spike filter (ATR(10))
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for ATR10 and volume MA
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10)
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate prior 1d OHLC for Camarilla levels (H3, L3)
    # We need to shift by 1 to use prior day's data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prior_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prior_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prior_close_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    # Camarilla H3 and L3: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_h3_1d = prior_close_1d + 1.1 * (prior_high_1d - prior_low_1d) / 6
    camarilla_l3_1d = prior_close_1d - 1.1 * (prior_high_1d - prior_low_1d) / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for ATR/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_slope[i]) or np.isnan(atr10_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA34 slope > 0 = uptrend (favor longs), < 0 = downtrend (favor shorts)
        uptrend = ema34_slope[i] > 0
        downtrend = ema34_slope[i] < 0
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: price < H3
            if position == 1:
                if curr_close < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i]:
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

name = "4h_Camarilla_H3L3_Breakout_12hEMA34Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0