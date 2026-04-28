#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; weekly trend ensures
# trades align with higher-timeframe momentum. Volume filters out false signals.
# Works in bull/bear: mean reversion in ranges, trend-following in strong moves.
# Target: 50-150 trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter
        vol_filter = volume[i] > vol_ma[i]
        
        # Weekly trend filter
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Williams %R levels
        oversold = willr[i] < -80
        overbought = willr[i] > -20
        
        # Entry conditions
        long_entry = oversold and vol_filter and trend_up
        short_entry = overbought and vol_filter and trend_down
        
        # Exit conditions: opposite extreme or trend change
        long_exit = (willr[i] > -20) or (not trend_up and position == 1)
        short_exit = (willr[i] < -80) or (not trend_down and position == -1)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_WeeklyTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0