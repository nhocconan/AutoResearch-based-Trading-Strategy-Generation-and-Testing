#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. 1w EMA50 ensures alignment with weekly trend.
# Volume confirmation filters false breakouts. Designed for low trade frequency (target: 7-25/year).
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for Donchian, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period high/low)
    # Use shift(1) to avoid look-ahead: based on previous 20 weekly candles
    high_roll = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume confirmation (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = df_1w['volume'].values > (1.5 * vol_ema_20)
    
    # Align 1w indicators to 1d timeframe
    high_roll_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(high_roll_aligned[i]) or np.isnan(low_roll_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian in uptrend with volume confirmation
            if high[i] > high_roll_aligned[i] and is_uptrend and volume_confirm_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian in downtrend with volume confirmation
            elif low[i] < low_roll_aligned[i] and is_downtrend and volume_confirm_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian (reversal) or hits upper Donchian (profit)
            if low[i] < low_roll_aligned[i] or high[i] > high_roll_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian (reversal) or hits lower Donchian (profit)
            if high[i] > high_roll_aligned[i] or low[i] < low_roll_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals