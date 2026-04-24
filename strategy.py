#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h Donchian trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h Donchian(20) for trend direction to capture major trend structure.
- EMA(9)/EMA(21) crossover on 1h for precise entry timing within the trend.
- Volume confirmation: current volume > 1.5 * 20-period average volume to ensure participation.
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective trend definition that works in both bull and bear markets.
- EMA crossover gives timely entries with reduced lag compared to longer MA pairs.
- Volume confirmation filters out breakouts lacking conviction.
- Estimated trades: ~80 total over 4 years (~20/year) based on EMA crossover frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period=20):
    """Calculate Donchian channels (upper and lower bands)."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    return highest_high.values, lowest_low.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter: Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 25:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    
    upper_4h, lower_4h = donchian_channels(df_4h['high'].values, df_4h['low'].values, 20)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Calculate 4h volume average for confirmation
    if len(df_4h) < 21:
        return np.zeros(n)
    
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # 1h EMA(9) and EMA(21)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    
    # 1h volume average for confirmation
    if len(close) < 21:
        return np.zeros(n)
    
    vol_ma_20_1h = pd.Series(close).rolling(window=20, min_periods=20).mean().values  # Using close as proxy for volume MA calculation
    # Actually calculate volume MA properly
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08:00-20:00 UTC)
    # prices.index is already DatetimeIndex with no conversion needed
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 25)  # Need EMA21 and Donchian data
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(vol_ma_20_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema9 = ema9[i]
        curr_ema21 = ema21[i]
        curr_ema9_prev = ema9[i-1] if i > 0 else curr_ema9
        curr_ema21_prev = ema21[i-1] if i > 0 else curr_ema21
        curr_upper_4h = upper_4h_aligned[i]
        curr_lower_4h = lower_4h_aligned[i]
        curr_vol_ma_20_4h = vol_ma_20_4h_aligned[i]
        curr_vol_ma_20_1h = vol_ma_20_1h[i]
        
        # EMA crossover signals
        ema9_above_ema21 = curr_ema9 > curr_ema21
        ema9_above_ema21_prev = curr_ema9_prev > curr_ema21_prev
        ema_bullish_cross = ema9_above_ema21 and not ema9_above_ema21_prev
        ema_bearish_cross = not ema9_above_ema21 and ema9_above_ema21_prev
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm_4h = curr_volume > 1.5 * curr_vol_ma_20_4h
        volume_confirm_1h = curr_volume > 1.5 * curr_vol_ma_20_1h
        volume_confirm = volume_confirm_4h and volume_confirm_1h
        
        # Trend filter: price relative to 4h Donchian channels
        price_above_upper_4h = curr_close > curr_upper_4h
        price_below_lower_4h = curr_close < curr_lower_4h
        price_in_channel = not (price_above_upper_4h or price_below_lower_4h)
        
        # Exit conditions: opposite EMA cross OR price breaks Donchian channel in opposite direction
        if position != 0:
            # Exit long: bearish EMA cross OR price breaks below lower 4h Donchian
            if position == 1:
                if ema_bearish_cross or price_below_lower_4h:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish EMA cross OR price breaks above upper 4h Donchian
            elif position == -1:
                if ema_bullish_cross or price_above_upper_4h:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: EMA cross with trend filter and volume confirmation
        if position == 0:
            # Long: bullish EMA cross AND price above lower 4h Donchian (bullish bias) AND volume confirmation
            if ema_bullish_cross and price_in_channel and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: bearish EMA cross AND price below upper 4h Donchian (bearish bias) AND volume confirmation
            elif ema_bearish_cross and price_in_channel and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_EMA9_21_Crossover_4hDonchian_TrendFilter_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0