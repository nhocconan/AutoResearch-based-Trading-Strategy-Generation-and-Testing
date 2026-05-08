#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w EMA10 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period high), price above 1w EMA10, and volume > 2x 20-period average.
# Short when price breaks below Donchian lower band (20-period low), price below 1w EMA10, and volume > 2x 20-period average.
# Exit when price crosses Donchian middle (20-period average of high/low) or trend fails.
# Donchian captures breakouts, EMA10 confirms higher timeframe trend, volume confirms strength.
# Designed to capture strong trends while avoiding false signals in ranging markets.
# Target: 12-37 trades/year to stay within profitable range.

name = "12h_Donchian_Breakout_1wEMA10_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(low_12h), np.nan)
    middle = np.full(len(close_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Calculate 12h 20-period average volume
    vol_ma_20 = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-20:i])
    
    # Get 1w data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA10 for trend filter
    ema_10 = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Calculate EMA10 slope for trend direction (rising/falling)
    ema_10_prev = np.roll(ema_10, 1)
    ema_10_prev[0] = ema_10[0]
    ema_10_rising = ema_10 > ema_10_prev
    ema_10_falling = ema_10 < ema_10_prev
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    ema_10_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_10_rising)
    ema_10_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_10_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_10_aligned[i]) or np.isnan(ema_10_rising_aligned[i]) or \
           np.isnan(ema_10_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Price breaks above/below Donchian bands + trend + volume
            # Long when price breaks above upper band, price above EMA10, with volume spike
            long_condition = (close[i] > upper_aligned[i]) and \
                             ema_10_rising_aligned[i] and (close[i] > ema_10_aligned[i]) and vol_filter
            # Short when price breaks below lower band, price below EMA10, with volume spike
            short_condition = (close[i] < lower_aligned[i]) and \
                              ema_10_falling_aligned[i] and (close[i] < ema_10_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle or trend fails
            if (close[i] < middle_aligned[i]) or (not ema_10_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle or trend fails
            if (close[i] > middle_aligned[i]) or (not ema_10_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals