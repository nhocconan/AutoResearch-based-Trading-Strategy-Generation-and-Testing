#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian Breakout with Weekly Trend and Volume Confirmation
# Hypothesis: Breakouts of daily Donchian channels (20-period) aligned with weekly trend
# (price above/below weekly 50 EMA) and volume confirmation work in both bull and bear markets.
# In bull markets: buy breakouts above upper band with weekly uptrend.
# In bear markets: sell breakdowns below lower band with weekly downtrend.
# Target: 10-25 trades/year (40-100 over 4 years).

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
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Weekly 50 EMA for trend filter
    weekly_close = df_weekly['close'].values
    weekly_close_series = pd.Series(weekly_close)
    ema_50_weekly = weekly_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    ema_50_weekly = np.roll(ema_50_weekly, 1)
    if len(ema_50_weekly) > 1:
        ema_50_weekly[0] = ema_50_weekly[1]
    else:
        ema_50_weekly[0] = 0
    
    # Align weekly EMA to daily timeframe
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily Donchian channels (20-period)
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
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA or Donchian lower band
            if close[i] < ema_50_weekly_aligned[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA or Donchian upper band
            if close[i] > ema_50_weekly_aligned[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above Donchian high with weekly uptrend and volume
            if (high[i] > donchian_high[i] and close[i] > donchian_high[i]) and \
               close[i] > ema_50_weekly_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below Donchian low with weekly downtrend and volume
            elif (low[i] < donchian_low[i] and close[i] < donchian_low[i]) and \
                 close[i] < ema_50_weekly_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals