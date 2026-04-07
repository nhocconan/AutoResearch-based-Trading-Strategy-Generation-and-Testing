#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume Confirmation
# Hypothesis: Price breaking out of weekly Donchian channels (20-period high/low)
# with volume confirmation and weekly trend filter (price vs weekly 50 EMA) captures
# strong momentum moves in both bull and bear markets.
# In bull markets: buy breakouts above weekly high with volume.
# In bear markets: sell breakouts below weekly low with volume.
# Weekly timeframe reduces noise, volume confirms institutional participation.
# Target: 8-15 trades/year (32-60 over 4 years).

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate rolling max/min for Donchian channels
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    donchian_high = weekly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed weekly bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Weekly trend filter: price vs 50 EMA
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema_50 = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema_50 = np.roll(weekly_ema_50, 1)
    if len(weekly_ema_50) > 1:
        weekly_ema_50[0] = weekly_ema_50[1]
    else:
        weekly_ema_50[0] = 0
    
    # Align weekly data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_50)
    
    # Volume filter: volume > 2.0x 50-period average (institutional participation)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(weekly_ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or close[i] < weekly_ema_50_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or close[i] > weekly_ema_50_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above weekly Donchian high with volume and trend filter
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                close[i] > weekly_ema_50_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly Donchian low with volume and trend filter
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  close[i] < weekly_ema_50_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals