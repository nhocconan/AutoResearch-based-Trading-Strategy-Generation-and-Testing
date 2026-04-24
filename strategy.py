#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend direction, 1d for volume spike filter (2.0x 20-period average volume).
- Camarilla levels: R3, S3, R4, S4 calculated from prior 1d OHLC.
- Entry: Long when price > R3 AND 12h EMA34 trending up AND 1d volume > 2.0 * 20-period average volume.
         Short when price < S3 AND 12h EMA34 trending down AND 1d volume > 2.0 * 20-period average volume.
- Exit: Price < R3 for long exit OR price > S3 for short exit.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading strong breaks in trending regimes with volume confirmation.
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
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate prior 1d OHLC for Camarilla levels (shifted by 1 to avoid look-ahead)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (yesterday's data for today's levels)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3, S3, R4, S4
    # R3 = prior_close + (prior_high - prior_low) * 1.1/4
    # S3 = prior_close - (prior_high - prior_low) * 1.1/4
    # R4 = prior_close + (prior_high - prior_low) * 1.1/2
    # S4 = prior_close - (prior_high - prior_low) * 1.1/2
    rang = prior_high - prior_low
    r3 = prior_close + rang * 1.1 / 4
    s3 = prior_close - rang * 1.1 / 4
    r4 = prior_close + rang * 1.1 / 2
    s4 = prior_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA34 slope (current > previous)
        if i > start_idx:
            ema34_trend_up = ema34_12h_aligned[i] > ema34_12h_aligned[i-1]
            ema34_trend_down = ema34_12h_aligned[i] < ema34_12h_aligned[i-1]
        else:
            ema34_trend_up = False
            ema34_trend_down = False
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit long: price < R3
            if position == 1:
                if curr_close < r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > S3
            elif position == -1:
                if curr_close > s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > R3 AND EMA34 trending up AND volume confirmation
            long_condition = (curr_close > r3_aligned[i] and 
                            ema34_trend_up and
                            volume_confirm)
            
            # Short: price < S3 AND EMA34 trending down AND volume confirmation
            short_condition = (curr_close < s3_aligned[i] and 
                             ema34_trend_down and
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

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0