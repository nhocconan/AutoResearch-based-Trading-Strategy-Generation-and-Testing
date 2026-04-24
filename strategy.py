#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA34 trend filter.
- Camarilla levels: calculated from prior 1d OHLC (R1, S1, R2, S2, R3, S3, R4, S4).
- Entry: Long when price closes above R1 AND price > 12h EMA34 AND volume > 2.0 * 20-period average volume.
         Short when price closes below S1 AND price < 12h EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Camarilla break (close below S1 for long, close above R1 for short) OR Camarilla width expansion (>1.5x average width).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla pivots work in both bull and bear markets as they identify key support/resistance levels from prior day's range.
- Volume confirmation ensures breakout legitimacy.
- 12h EMA34 provides medium-term trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for the day.
    Based on prior day's OHLC.
    Returns: R4, R3, R2, R1, PP, S1, S2, S3, S4
    """
    range_val = high - low
    if range_val == 0:
        return np.full(9, close)  # Avoid division by zero
    
    pp = (high + low + close) / 3
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s4 = close - range_val * 1.1 / 2
    
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h volume average for confirmation
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate Camarilla levels from prior 1d data (need to shift by 1 to avoid look-ahead)
    # We'll calculate daily levels and align them to 4h bars
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    camarilla_data = []
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = camarilla_levels(h, l, c)
        camarilla_data.append([r4, r3, r2, r1, pp, s1, s2, s3, s4])
    
    camarilla_array = np.array(camarilla_data)
    
    # Align each Camarilla level to 4h timeframe
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 0])  # R4
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 1])  # R3
    r2_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 2])  # R2
    r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 3])  # R1
    pp_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 4])  # PP
    s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 5])  # S1
    s2_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 6])  # S2
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 7])  # S3
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_array[:, 8])  # S4
    
    # Calculate average Camarilla width (R1-S1) for expansion filter
    camarilla_width = r1_1d - s1_1d
    # Calculate 50th percentile of width for expansion condition (using expanding window)
    camarilla_width_percentile = np.zeros_like(camarilla_width)
    for i in range(len(camarilla_width)):
        if i < 20:
            camarilla_width_percentile[i] = np.nan
        else:
            camarilla_width_percentile[i] = np.percentile(camarilla_width[max(0, i-50):i+1], 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need 34 for 12h EMA, 20 for volume MA, 1 for daily data
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(camarilla_width[i]) or
            np.isnan(camarilla_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price closes below S1 OR Camarilla width expands above 1.5x 50th percentile
            if position == 1:
                if curr_close < s1_1d[i] or camarilla_width[i] > camarilla_width_percentile[i] * 1.5:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price closes above R1 OR Camarilla width expands above 1.5x 50th percentile
            elif position == -1:
                if curr_close > r1_1d[i] or camarilla_width[i] > camarilla_width_percentile[i] * 1.5:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Camarilla breakout signals (close-based)
            breakout_up = curr_close > r1_1d[i] and prev_close <= r1_1d[i-1]
            breakout_down = curr_close < s1_1d[i] and prev_close >= s1_1d[i-1]
            
            # Trend filter: price vs 12h EMA34
            long_trend = curr_close > ema34_12h_aligned[i]
            short_trend = curr_close < ema34_12h_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 2.0 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and long_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif breakout_down and short_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0