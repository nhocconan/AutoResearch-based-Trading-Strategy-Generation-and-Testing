#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: On 4h timeframe, buy when price breaks above Donchian(20) high with 1d uptrend and volume > 1.5x average.
# Sell when price breaks below Donchian(20) low with 1d downtrend and volume > 1.5x average.
# Exit when price touches opposite Donchian band or volume drops below average.
# Uses 1d trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures momentum behind breakouts.
# Target: 20-50 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(daily_ema50_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band or volume drops below average
            if close[i] <= donchian_low[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band or volume drops below average
            if close[i] >= donchian_high[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # 1d trend filter
            daily_uptrend = close[i] > daily_ema50_4h[i]
            daily_downtrend = close[i] < daily_ema50_4h[i]
            
            # Long entry: price breaks above upper Donchian band with volume and uptrend
            if close[i] > donchian_high[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with volume and downtrend
            elif close[i] < donchian_low[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals