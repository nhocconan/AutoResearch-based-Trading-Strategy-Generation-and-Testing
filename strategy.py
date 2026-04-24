#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter and volume average.
- Camarilla pivots: calculates R3, R2, R1, S1, S2, S3 from prior day's OHLC.
- Entry: Long when price breaks above R3 AND price > 4h EMA34 AND volume > 1.5 * 20-period average volume.
         Short when price breaks below S3 AND price < 4h EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (R1 for long, S1 for short) or 4h EMA34 cross.
- Signal size: 0.20 discrete to minimize fee drag.
- Works in bull/bear: Camarilla levels adapt to volatility; 4h EMA34 filters counter-trend trades.
- Volume confirmation ensures breakout legitimacy.
- Uses session filter (08-20 UTC) to reduce noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_pivots(high, low, close):
    """
    Calculate Camarilla pivot levels for the day.
    Based on prior day's high, low, close.
    Returns: R4, R3, R2, R1, PP, S1, S2, S3, S4
    We use R3, R1, S1, S3 for trading.
    """
    range_ = high - low
    if range_ == 0:
        return np.array([close]*9)  # Avoid division by zero
    
    pp = (high + low + close) / 3
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)
    s4 = close - (range_ * 1.1 / 2)
    
    return np.array([r4, r3, r2, r1, pp, s1, s2, s3, s4])

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_4h = ema(df_4h['close'].values, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 4h volume average for confirmation
    if len(df_4h) < 20:
        return np.zeros(n)
    
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate Camarilla pivots from 1d data (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get prior day's OHLC for each 1h bar
    # We need to shift 1d data by 1 to avoid look-ahead (use prior day's close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_data = np.zeros((len(high_1d), 9))  # [R4, R3, R2, R1, PP, S1, S2, S3, S4]
    for i in range(len(high_1d)):
        camarilla_data[i] = camarilla_pivots(high_1d[i], low_1d[i], close_1d[i])
    
    # Align Camarilla levels to 1h timeframe (use prior day's levels)
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_data[:, 1])  # R3 is index 1
    r1_1d = align_htf_to_ltf(prices, df_1d, camarilla_data[:, 3])  # R1 is index 3
    s1_1d = align_htf_to_ltf(prices, df_1d, camarilla_data[:, 5])  # S1 is index 5
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_data[:, 7])  # S3 is index 7
    
    # Precompute session filter (08-20 UTC)
    # open_time is already datetime64[ns]
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need 34 for 4h EMA, 20 for volume MA, 1 for prior day data
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or
            np.isnan(r3_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(s3_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below R1 OR price < 4h EMA34
            if position == 1:
                if curr_close < r1_1d[i] or curr_close < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above S1 OR price > 4h EMA34
            elif position == -1:
                if curr_close > s1_1d[i] or curr_close > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_close > r3_1d[i] and prev_close <= r3_1d[i]
            breakout_down = curr_close < s3_1d[i] and prev_close >= s3_1d[i]
            
            # Trend filter: price vs 4h EMA34
            long_trend = curr_close > ema34_4h_aligned[i]
            short_trend = curr_close < ema34_4h_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_4h_aligned[i] if not np.isnan(vol_ma_20_4h_aligned[i]) else False
            
            if breakout_up and long_trend and volume_confirm:
                signals[i] = 0.20
                position = 1
            elif breakout_down and short_trend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0