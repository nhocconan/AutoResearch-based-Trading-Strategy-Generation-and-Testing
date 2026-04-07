#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Range Breakout with Volume and Trend Filter
# Hypothesis: Price breaking above/below the previous day's high/low on 12h timeframe
# indicates continuation of the previous day's trend. Volume confirms institutional participation.
# Trend filter (price above/below 200 EMA) ensures alignment with higher timeframe trend.
# Works in both bull and bear markets: in bull, only long breakouts; in bear, only short breakdowns.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_daily_range_breakout_volume_trend_v1"
timeframe = "12h"
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
    
    # Get daily data for range calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    
    # Align to 12h timeframe (use previous day's levels)
    daily_high_aligned = align_htf_to_ltf(prices, df_daily, prev_daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_daily, prev_daily_low)
    
    # 1d trend filter: price above/below 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below previous day's low or trend turns bearish or volume drops
            if (low[i] < daily_low_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above previous day's high or trend turns bullish or volume drops
            if (high[i] > daily_high_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above previous day's high with volume and bullish trend
            if ((high[i] > daily_high_aligned[i] or close[i] > daily_high_aligned[i]) and 
                close[i] > ema_200[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below previous day's low with volume and bearish trend
            elif ((low[i] < daily_low_aligned[i] or close[i] < daily_low_aligned[i]) and 
                  close[i] < ema_200[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals