#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend direction.
- Entry: Long when price breaks above Camarilla R3 level AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below Camarilla S3 level AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite breakout (price closes below R3 for long exit, above S3 for short exit) OR trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide precise intraday support/resistance; breakouts with volume confirmation capture strong moves.
- Works in bull markets (buying breakouts) and bear markets (selling breakdowns) with trend filter avoiding counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 4.0)
    s3 = pivot - (range_ * 1.1 / 4.0)
    return r3, s3

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
    if len(df_12h) < 34:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate daily Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    camarilla_high_1d = df_1d['high'].values
    camarilla_low_1d = df_1d['low'].values
    camarilla_close_1d = df_1d['close'].values
    
    r3_1d, s3_1d = calculate_camarilla(camarilla_high_1d, camarilla_low_1d, camarilla_close_1d)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price closes below R3 OR trend reversal (price < EMA34)
            if position == 1:
                if curr_close < r3_aligned[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price closes above S3 OR trend reversal (price > EMA34)
            elif position == -1:
                if curr_close > s3_aligned[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            # Long: price breaks above R3 AND price > 12h EMA34 AND volume confirmation
            long_condition = (curr_high > r3_aligned[i] and 
                            curr_close > ema34_12h_aligned[i] and
                            volume_confirm)
            
            # Short: price breaks below S3 AND price < 12h EMA34 AND volume confirmation
            short_condition = (curr_low < s3_aligned[i] and 
                             curr_close < ema34_12h_aligned[i] and
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