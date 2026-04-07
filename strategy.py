#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Monthly Donchian Breakout with Volume and Trend Filter
# Hypothesis: Price breaking out of monthly Donchian channels (20-period high/low)
# with volume confirmation (>1.5x 50-period average volume) and monthly trend filter
# (price vs monthly 50 EMA) captures strong momentum moves. Monthly timeframe
# reduces noise and captures major trends in both bull and bear markets.
# Target: 15-25 trades/year (60-100 over 4 years) to avoid fee drag.

name = "4h_monthly_donchian_breakout_volume_trend_v1"
timeframe = "4h"
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
    
    # Get monthly data for Donchian channels and trend filter
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) < 50:
        return np.zeros(n)
    
    # Calculate monthly Donchian channels (20-period high/low)
    monthly_high = df_monthly['high'].values
    monthly_low = df_monthly['low'].values
    monthly_close = df_monthly['close'].values
    
    # Calculate rolling max/min for Donchian channels
    monthly_high_series = pd.Series(monthly_high)
    monthly_low_series = pd.Series(monthly_low)
    donchian_high = monthly_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = monthly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed monthly bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Monthly trend filter: price vs 50 EMA
    monthly_close_series = pd.Series(monthly_close)
    monthly_ema_50 = monthly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    monthly_ema_50 = np.roll(monthly_ema_50, 1)
    if len(monthly_ema_50) > 1:
        monthly_ema_50[0] = monthly_ema_50[1]
    else:
        monthly_ema_50[0] = 0
    
    # Align monthly data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_monthly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_monthly, donchian_low)
    monthly_ema_50_aligned = align_htf_to_ltf(prices, df_monthly, monthly_ema_50)
    
    # Volume filter: volume > 1.5x 50-period average (institutional participation)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(monthly_ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below monthly Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or close[i] < monthly_ema_50_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above monthly Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or close[i] > monthly_ema_50_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above monthly Donchian high with volume and trend filter
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                close[i] > monthly_ema_50_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below monthly Donchian low with volume and trend filter
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  close[i] < monthly_ema_50_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals