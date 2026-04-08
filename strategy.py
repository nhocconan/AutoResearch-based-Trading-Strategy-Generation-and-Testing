#!/usr/bin/env python3
# 4h_pullback_breakout_1d_trend_volume_v1
# Hypothesis: 4h breakout of 20-period high/low with volume > 1.5x 20-period average, in direction of daily trend (price above/below daily EMA50).
# Pullback entries: wait for pullback to 20-period EMA after breakout to avoid chasing.
# Works in bull markets by capturing continuation breakouts and in bear markets by capturing breakdowns.
# Volume filter ensures participation, trend filter avoids counter-trend trades, pullback reduces false breakouts.
# Target: 20-40 trades/year with ~0.25 position size to minimize fee drag.

name = "4h_pullback_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout
    high_20 = np.full_like(high, np.nan)
    low_20 = np.full_like(low, np.nan)
    for i in range(19, n):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    # 20-period EMA for pullback entries
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume moving average (20-period)
    vol_ma = np.full_like(volume, np.nan)
    vol_series = pd.Series(volume)
    vol_ma[19:] = vol_series.rolling(window=20, min_periods=20).mean().values[:n-19]
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for trend filter
    ema_period = 50
    ema_daily = np.full_like(close_daily, np.nan)
    ema_daily[ema_period-1:] = pd.Series(close_daily).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values[ema_period-1:]
    
    # Align daily EMA to 4h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    breakout_high = False  # Track if we've had a recent bullish breakout
    breakout_low = False   # Track if we've had a recent bearish breakout
    
    # Start from sufficient lookback
    start_idx = max(20, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            # Reset breakout flags when out of session
            breakout_high = False
            breakout_low = False
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or volume[i] == 0 or 
            np.isnan(ema_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or volume fails or price drops below EMA20
            if not uptrend_htf or not volume_filter or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
                breakout_high = False  # Reset breakout flag on exit
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or volume fails or price rises above EMA20
            if not downtrend_htf or not volume_filter or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
                breakout_low = False  # Reset breakout flag on exit
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Check for breakout conditions
            bullish_breakout = (close[i] > high_20[i-1]) and volume_filter and uptrend_htf
            bearish_breakout = (close[i] < low_20[i-1]) and volume_filter and downtrend_htf
            
            # Long entry: bullish breakout OR pullback to EMA20 after bullish breakout
            if bullish_breakout or (breakout_high and close[i] >= ema_20[i] and close[i] > low[i-1]):
                position = 1
                signals[i] = 0.25
                breakout_high = True  # Set breakout flag on entry
                breakout_low = False  # Reset opposite flag
            # Short entry: bearish breakout OR pullback to EMA20 after bearish breakout
            elif bearish_breakout or (breakout_low and close[i] <= ema_20[i] and close[i] < high[i-1]):
                position = -1
                signals[i] = -0.25
                breakout_low = True  # Set breakout flag on entry
                breakout_high = False  # Reset opposite flag
    
    return signals