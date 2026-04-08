#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: On 12h timeframe, use daily Donchian breakout with daily trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and daily trend up.
# Short when price breaks below Donchian(20) low with volume > 1.5x average and daily trend down.
# Exit on opposite Donchian break or when volume drops below average.
# Daily trend defined by price above/below daily EMA20.
# This strategy targets fewer trades (12-37/year) by using higher timeframe structure and tight entry conditions.
# Works in both bull and bear markets via trend filter and volatility-based breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Calculate Donchian channels from daily data (using previous day's data)
    # Get daily data
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels using previous day's data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Donchian(20) using previous 20 days
    donchian_high = np.zeros_like(daily_high)
    donchian_low = np.zeros_like(daily_low)
    
    for i in range(len(daily_high)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(daily_high[i-20:i])
            donchian_low[i] = np.min(daily_low[i-20:i])
    
    # Align daily Donchian levels to 12h timeframe (with proper delay for daily bar close)
    donchian_high_12h = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # Daily trend filter: price above/below daily EMA20
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_12h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 12h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or np.isnan(daily_ema20_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or volume drops below average
            if close[i] < donchian_low_12h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or volume drops below average
            if close[i] > donchian_high_12h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_12h[i]
            daily_downtrend = close[i] < daily_ema20_12h[i]
            
            # Long entry: price breaks above Donchian high with volume and uptrend
            if close[i] > donchian_high_12h[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and downtrend
            elif close[i] < donchian_low_12h[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals