#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period high), price above 1d EMA34, and volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20-period low), price below 1d EMA34, and volume > 1.5x 20-period average.
# Exit when price crosses Donchian middle (20-period average of high/low) or trend fails.
# Designed to capture strong trends while avoiding false signals in ranging markets.
# Target: 20-50 trades/year to stay within profitable range.

name = "4h_Donchian_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper = np.full(len(high_4h), np.nan)
    lower = np.full(len(low_4h), np.nan)
    middle = np.full(len(close_4h), np.nan)
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Calculate 4h 20-period average volume
    vol_ma_20 = np.full(len(volume_4h), np.nan)
    for i in range(20, len(volume_4h)):
        vol_ma_20[i] = np.mean(volume_4h[i-20:i])
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Align all indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
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
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 4h bar's volume
            idx_4h = 0
            while idx_4h < len(df_4h) and df_4h.iloc[idx_4h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_4h += 1
            idx_4h -= 1  # last completed 4h bar
            
            if idx_4h >= 0:
                vol_4h_current = df_4h.iloc[idx_4h]['volume']
                vol_filter = vol_4h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Price breaks above/below Donchian bands + trend + volume
            # Long when price breaks above upper band, price above EMA34, with volume spike
            long_condition = (close[i] > upper_aligned[i]) and \
                             ema_34_rising_aligned[i] and (close[i] > ema_34_aligned[i]) and vol_filter
            # Short when price breaks below lower band, price below EMA34, with volume spike
            short_condition = (close[i] < lower_aligned[i]) and \
                              ema_34_falling_aligned[i] and (close[i] < ema_34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian middle or trend fails
            if (close[i] < middle_aligned[i]) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian middle or trend fails
            if (close[i] > middle_aligned[i]) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals