#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout of Camarilla R1/S1 levels on 4h with 1d trend filter (EMA34) and volume confirmation.
# In bull markets: buy R1 breakout when price > 1d EMA34 and volume > 1.5x average.
# In bear markets: short S1 breakdown when price < 1d EMA34 and volume > 1.5x average.
# Uses daily trend to avoid counter-trend trades, volume to confirm breakout strength.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume (20-period) for volume confirmation
    vol_series = pd.Series(volume)
    avg_vol_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume average
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_vol_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 4h bar using daily OHLC
        # Camarilla: based on previous day's range
        if i == 0:
            # Need previous day's data - use available data
            prev_high = high_1d[0] if len(high_1d) > 0 else high[i]
            prev_low = low_1d[0] if len(low_1d) > 0 else low[i]
            prev_close = close_1d[0] if len(close_1d) > 0 else close[i]
        else:
            # Get previous day's data - careful with indexing
            day_idx = i // 6  # Approximate: 6 four-hour bars per day
            if day_idx > 0 and day_idx < len(high_1d):
                prev_high = high_1d[day_idx - 1]
                prev_low = low_1d[day_idx - 1]
                prev_close = close_1d[day_idx - 1]
            else:
                # Fallback to first available day
                prev_high = high_1d[0]
                prev_low = low_1d[0]
                prev_close = close_1d[0]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            # Skip if invalid range
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Camarilla R1 and S1 levels
        r1 = prev_close + (range_val * 1.1 / 12)
        s1 = prev_close - (range_val * 1.1 / 12)
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        # Trend filter: price vs 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R1 with volume and uptrend
            if close[i] > r1 and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume and downtrend
            elif close[i] < s1 and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below S1 or trend turns down
            if close[i] < s1 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or trend turns up
            if close[i] > r1 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals