#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Hypothesis: Donchian breakouts capture strong trends; weekly trend filter avoids counter-trend trades; volume confirms institutional interest.
# Works in bull via upward breakouts, in bear via downward breakdowns. Weekly trend filter reduces whipsaws in ranging markets.
# Target: 15-25 trades/year to minimize fee drag on daily timeframe.
name = "daily_donchian20_weekly_trend_volume_v1"
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
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 20-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Donchian channels (20-period high/low)
    # Using rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume 20-period moving average for confirmation
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA (trend reversal)
            if close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA (trend reversal)
            if close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian high + volume confirmation + weekly uptrend
            if (close[i] > donchian_high[i] and vol_confirm and 
                close[i] > ema_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low + volume confirmation + weekly downtrend
            elif (close[i] < donchian_low[i] and vol_confirm and 
                  close[i] < ema_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals