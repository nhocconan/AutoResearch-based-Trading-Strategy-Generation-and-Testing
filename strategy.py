#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
- Entry: Price breaks above/below 20-day Donchian channel
- Trend filter: Price above/below 200-week EMA (bullish if close > EMA, bearish if close < EMA)
- Volume: Daily volume > 1.5x 20-day average to confirm breakout
- Exit: Opposite Donchian breakout or trend reversal
- Position sizing: 0.25 for long, -0.25 for short
Target: 10-25 trades/year (40-100 total over 4 years) - low frequency to minimize fee drag
Works in bull markets via breakouts, works in bear via shorting breakdowns
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 200-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_1w_up = close_1w > ema_200_1w
    trend_1w_down = close_1w < ema_200_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price breaks below lower Donchian OR weekly trend turns down
            if (close[i] < low_20[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian OR weekly trend turns up
            if (close[i] > high_20[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price breaks above upper Donchian + weekly uptrend + volume
            if (close[i] > high_20[i]) and trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower Donchian + weekly downtrend + volume
            elif (close[i] < low_20[i]) and trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals