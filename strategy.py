#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3, 12h EMA34 up-trend, volume > 2.5x average
# Short when price breaks below Camarilla S3, 12h EMA34 down-trend, volume > 2.5x average
# Exit when price crosses the Camarilla pivot point (mean reversion)
# Uses discrete position sizing (0.25) and strong filters to target 20-50 trades/year.
# Designed to work in both bull and bear markets by following higher timeframe trend.

name = "4h_Camarilla_R3S3_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation (previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous 4h day's OHLC (24 periods = 1 day)
    # We need to calculate for each 4h bar using the prior day's OHLC
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    open_4h = df_4h['open'].values
    
    # Calculate prior day's OHLC (24 * 4h bars = 1 day)
    # For each point, we use OHLC from 24 bars ago
    lookback = 24  # 24 * 4h = 96h = 4 days? Wait, 6 * 4h = 24h = 1 day
    lookback = 6   # 6 * 4h bars = 24h = 1 day
    
    # Shift by lookback to get prior day's OHLC
    prior_close = np.roll(close_4h, lookback)
    prior_high = np.roll(high_4h, lookback)
    prior_low = np.roll(low_4h, lookback)
    prior_open = np.roll(open_4h, lookback)
    
    # Set first 'lookback' values to NaN (no prior day data)
    prior_close[:lookback] = np.nan
    prior_high[:lookback] = np.nan
    prior_low[:lookback] = np.nan
    prior_open[:lookback] = np.nan
    
    # Calculate Camarilla levels
    # Pivot = (Prior High + Prior Low + Prior Close) / 3
    pivot = (prior_high + prior_low + prior_close) / 3.0
    # Range = Prior High - Prior Low
    rng = prior_high - prior_low
    
    # Camarilla levels
    R3 = pivot + (rng * 1.1 / 4)
    S3 = pivot - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 34, 20)  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_R3 = R3_aligned[i]
        curr_S3 = S3_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_ema34_12h = ema_34_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below pivot point (mean reversion)
            if curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above pivot point (mean reversion)
            if curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.5x 20-period average (strong filter)
            vol_confirmed = curr_volume > 2.5 * curr_vol_ma
            
            # Long when price breaks above R3, 12h EMA34 up-trend, volume confirmed
            if curr_high > curr_R3 and curr_close > curr_ema34_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3, 12h EMA34 down-trend, volume confirmed
            elif curr_low < curr_S3 and curr_close < curr_ema34_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals