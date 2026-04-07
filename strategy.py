#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend and Volume Filter
# Hypothesis: Daily Donchian(20) breakouts capture strong momentum. Weekly trend (via EMA50) filters direction, and volume confirms institutional participation.
# Works in bull markets: breaks above upper band with weekly uptrend and volume = continuation.
# Works in bear markets: breaks below lower band with weekly downtrend and volume = continuation.
# Uses discrete position sizing (0.25) to limit trade frequency and fee drag.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "daily_donchian_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly EMA50 or Donchian low or volume drops
            if (close[i] <= weekly_ema50_aligned[i] or close[i] <= donchian_low[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly EMA50 or Donchian high or volume drops
            if (close[i] >= weekly_ema50_aligned[i] or close[i] >= donchian_high[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with weekly uptrend and volume
            if (high[i] > donchian_high[i] and close[i] > donchian_high[i] and 
                close[i] > weekly_ema50_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with weekly downtrend and volume
            elif (low[i] < donchian_low[i] and close[i] < donchian_low[i] and 
                  close[i] < weekly_ema50_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals