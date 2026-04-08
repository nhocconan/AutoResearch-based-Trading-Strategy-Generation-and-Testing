#!/usr/bin/env python3
# 12h_donchian_breakout_1w_trend_volume_v1
# Hypothesis: On 12h timeframe, use weekly Donchian channel breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above weekly Donchian high (20) with volume > 1.5x average and weekly trend up.
# Short when price breaks below weekly Donchian low (20) with volume > 1.5x average and weekly trend down.
# Exit when price returns to weekly Donchian midpoint or volume drops below average.
# Weekly trend defined by price above/below weekly EMA20.
# This strategy targets fewer trades (12-37/year) by using higher timeframe structure and tight entry conditions.
# Works in both bull and bear markets via trend filter and mean reversion in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
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
    
    # Get weekly data for Donchian channels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly high, low, close
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Using previous week's data to avoid look-ahead
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_12h = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Weekly trend filter: price above/below weekly EMA20
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().shift(1).values
    weekly_ema20_12h = align_htf_to_ltf(prices, df_weekly, weekly_ema20)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or np.isnan(donchian_mid_12h[i]) or np.isnan(weekly_ema20_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or volume drops below average
            if close[i] <= donchian_mid_12h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or volume drops below average
            if close[i] >= donchian_mid_12h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Weekly trend filter
            weekly_uptrend = close[i] > weekly_ema20_12h[i]
            weekly_downtrend = close[i] < weekly_ema20_12h[i]
            
            # Long entry: price breaks above weekly Donchian high with volume and uptrend
            if close[i] > donchian_high_12h[i] and volume_ok and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume and downtrend
            elif close[i] < donchian_low_12h[i] and volume_ok and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals