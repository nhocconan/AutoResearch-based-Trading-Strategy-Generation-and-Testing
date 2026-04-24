#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 75-150 total trades over 4 years (19-38/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Camarilla levels from 1d: R3, S3, R4, S4 calculated from prior 1d OHLC.
- Trend: price > EMA50(1d) for long bias, price < EMA50(1d) for short bias.
- Volume: current 6h volume > 2.0 * 20-period average 6h volume.
- Entry: Long when price > R3 AND price > EMA50 AND volume confirmation.
         Short when price < S3 AND price < EMA50 AND volume confirmation.
- Exit: Opposite Camarilla break (price < R3 for long exit, price > S3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via breakouts with trend, avoids whipsaws in bear via trend filter.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1d Camarilla levels (R3, S3, R4, S4) from prior 1d OHLC
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using prior day's OHLC (shifted by 1 to avoid look-ahead)
    rng = high_1d - low_1d
    r4 = close_1d + 1.1 * rng * 1.1 / 2
    r3 = close_1d + 1.1 * rng * 1.1 / 4
    s3 = close_1d - 1.1 * rng * 1.1 / 4
    s4 = close_1d - 1.1 * rng * 1.1 / 2
    
    # Shift by 1 to use prior day's levels (avoid look-ahead)
    r4 = np.concatenate([[np.nan], r4[:-1]])
    r3 = np.concatenate([[np.nan], r3[:-1]])
    s3 = np.concatenate([[np.nan], s3[:-1]])
    s4 = np.concatenate([[np.nan], s4[:-1]])
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_1d_aligned[i]
        short_bias = curr_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla break (R3/S3)
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
            # Long: price > R3 AND long bias AND volume confirmation
            long_condition = (curr_close > r3_aligned[i] and 
                            long_bias and
                            volume_confirm)
            
            # Short: price < S3 AND short bias AND volume confirmation
            short_condition = (curr_close < s3_aligned[i] and 
                             short_bias and
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

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0