#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 60-period Donchian high, price above 1d EMA50, and volume > 1.8x 20-period average.
# Short when price breaks below 60-period Donchian low, price below 1d EMA50, and volume > 1.8x 20-period average.
# Exit when price crosses the 60-period Donchian midpoint or trend fails.
# Donchian channels provide clear breakout levels; EMA50 filters trend direction; volume confirms breakout strength.
# Designed to capture strong trending moves while avoiding false breakouts in choppy markets.
# Target: 15-30 trades/year to stay within profitable range.

name = "6h_Donchian_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 60-period Donchian from 6h data
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 60-period Donchian channels
    donchian_high = pd.Series(high_6h).rolling(window=60, min_periods=60).max().values
    donchian_low = pd.Series(low_6h).rolling(window=60, min_periods=60).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 6h 20-period average volume for volume filter
    vol_ma_20 = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate EMA50 slope for trend direction (rising/falling)
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = ema_50[0]
    ema_50_rising = ema_50 > ema_50_prev
    ema_50_falling = ema_50 < ema_50_prev
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
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
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_rising_aligned[i]) or \
           np.isnan(ema_50_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.8x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 6h bar's volume
            idx_6h = 0
            while idx_6h < len(df_6h) and df_6h.iloc[idx_6h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_6h += 1
            idx_6h -= 1  # last completed 6h bar
            
            if idx_6h >= 0:
                vol_6h_current = df_6h.iloc[idx_6h]['volume']
                vol_filter = vol_6h_current > 1.8 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            # Long when price breaks above Donchian high, price above EMA50, with volume spike
            long_condition = (close[i] > donchian_high_aligned[i]) and \
                             ema_50_rising_aligned[i] and (close[i] > ema_50_aligned[i]) and vol_filter
            # Short when price breaks below Donchian low, price below EMA50, with volume spike
            short_condition = (close[i] < donchian_low_aligned[i]) and \
                              ema_50_falling_aligned[i] and (close[i] < ema_50_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses Donchian midpoint or trend fails
            if (close[i] < donchian_mid_aligned[i]) or (not ema_50_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses Donchian midpoint or trend fails
            if (close[i] > donchian_mid_aligned[i]) or (not ema_50_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals