#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and Trend Filter
# Hypothesis: Daily Donchian channels (20-period) act as significant support/resistance.
# Breakouts above upper channel with volume and trend confirmation indicate bullish continuation.
# Breakdowns below lower channel with volume and trend confirmation indicate bearish continuation.
# Uses 1d trend filter (price above/below 50 EMA) to avoid counter-trend trades.
# Volume filter ensures institutional participation. Works in bull/bear markets by
# aligning with trend: in bull, only long breakouts; in bear, only short breakdowns.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "12h_daily_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Upper channel: highest high over past 20 days
    upper_channel = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 days
    lower_channel = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (use previous day's levels to avoid look-ahead)
    upper_channel_aligned = align_htf_to_ltf(prices, df_daily, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_daily, lower_channel)
    
    # 1d trend filter: price above/below 50 EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below lower channel or trend turns bearish or volume drops
            if (close[i] < lower_channel_aligned[i] or close[i] < ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above upper channel or trend turns bullish or volume drops
            if (close[i] > upper_channel_aligned[i] or close[i] > ema_50[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above upper channel with volume and bullish trend
            if ((high[i] > upper_channel_aligned[i] or close[i] > upper_channel_aligned[i]) and 
                close[i] > ema_50[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower channel with volume and bearish trend
            elif ((low[i] < lower_channel_aligned[i] or close[i] < lower_channel_aligned[i]) and 
                  close[i] < ema_50[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals