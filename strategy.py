#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In trending markets (EMA34 slope), we go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# Volume confirmation ensures institutional participation. Works in bull markets (riding uptrends) and bear markets (riding downtrends).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_ElderRay_Trend_Volume"
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
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_6h - ema13  # High - EMA13
    bear_power = low_6h - ema13   # Low - EMA13
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Calculate EMA34 slope for trend direction
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    ema_34_rising = ema_34 > ema_34_prev
    ema_34_falling = ema_34 < ema_34_prev
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_6h), np.nan)
    for i in range(20, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-20:i])
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
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
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_rising_aligned[i]) or \
           np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 6h bar's volume
            idx_6h = 0
            while idx_6h < len(df_6h) and df_6h.iloc[idx_6h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_6h += 1
            idx_6h -= 1  # last completed 6h bar
            
            if idx_6h >= 0:
                vol_6h_current = df_6h.iloc[idx_6h]['volume']
                vol_filter = vol_6h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Elder Ray signals + trend + volume confirmation
            # Long when Bull Power > 0 and rising in uptrend (EMA34 rising) with volume
            long_condition = (bull_power_aligned[i] > 0) and \
                             (bull_power_aligned[i] > bull_power_aligned[i-1]) and \
                             ema_34_rising_aligned[i] and vol_filter
            # Short when Bear Power < 0 and falling in downtrend (EMA34 falling) with volume
            short_condition = (bear_power_aligned[i] < 0) and \
                              (bear_power_aligned[i] < bear_power_aligned[i-1]) and \
                              ema_34_falling_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or trend fails
            if (bull_power_aligned[i] <= 0) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or trend fails
            if (bear_power_aligned[i] >= 0) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals