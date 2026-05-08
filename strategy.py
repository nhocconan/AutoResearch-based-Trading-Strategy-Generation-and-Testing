#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (EMA34 slope),
# we fade extreme readings (>80 for short, <20 for long) only when volume confirms institutional interest.
# Works in bull markets (fading overextended rallies) and bear markets (fading oversold bounces).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

name = "6h_WilliamsR_MeanRev_1dEMA34_Volume"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R (14-period): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.full(len(high_6h), np.nan)
    lowest_low = np.full(len(low_6h), np.nan)
    for i in range(14, len(high_6h)):
        highest_high[i] = np.max(high_6h[i-13:i+1])
        lowest_low[i] = np.min(low_6h[i-13:i+1])
    
    williams_r = np.full(len(close_6h), np.nan)
    for i in range(14, len(close_6h)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = ((highest_high[i] - close_6h[i]) / (highest_high[i] - lowest_low[i])) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
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
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
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
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or \
           np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]):
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
            # Look for entry: Williams %R extremes + trend + volume confirmation
            # Long when oversold (< -80) in uptrend (EMA34 rising) with volume
            long_condition = (williams_r_aligned[i] < -80) and \
                             ema_34_rising_aligned[i] and vol_filter
            # Short when overbought (> -20) in downtrend (EMA34 falling) with volume
            short_condition = (williams_r_aligned[i] > -20) and \
                              ema_34_falling_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend fails
            if (williams_r_aligned[i] > -50) or (not ema_34_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend fails
            if (williams_r_aligned[i] < -50) or (not ema_34_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals