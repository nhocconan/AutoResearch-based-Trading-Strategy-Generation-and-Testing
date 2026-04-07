#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Weekly Range Breakout with Volume and Trend Filter
# Hypothesis: Price breaking above/below the previous week's high/low indicates
# continuation of the previous week's trend. Volume confirms institutional participation.
# Trend filter (price above/below 200 EMA) ensures alignment with higher timeframe trend.
# Works in both bull and bear markets: in bull, only long breakouts; in bear, only short breakdowns.
# Target: 12-37 trades/year (50-150 over 4 years).

name = "12h_weekly_range_breakout_volume_trend_v1"
timeframe = "12h"
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
    
    # Get weekly data for range calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    
    # Align to 12h timeframe (use previous week's levels)
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_low)
    
    # 1w trend filter: price above/below 200 EMA (using weekly close)
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
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below previous week's low or trend turns bearish or volume drops
            if (low[i] < weekly_low_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above previous week's high or trend turns bullish or volume drops
            if (high[i] > weekly_high_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above previous week's high with volume and bullish trend
            if ((high[i] > weekly_high_aligned[i] or close[i] > weekly_high_aligned[i]) and 
                close[i] > ema_200[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below previous week's low with volume and bearish trend
            elif ((low[i] < weekly_low_aligned[i] or close[i] < weekly_low_aligned[i]) and 
                  close[i] < ema_200[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals