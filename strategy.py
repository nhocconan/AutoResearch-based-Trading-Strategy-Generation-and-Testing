#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: Use 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.5x average and price > 12h EMA50.
# Short when price breaks below 20-period low with volume > 1.5x average and price < 12h EMA50.
# Exit on opposite Donchian breakout or when price crosses the 20-period midpoint.
# Designed to capture trend moves with confirmation to reduce false signals.
# Target: 20-40 trades/year to minimize fee drag while capturing meaningful trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(donchian_mid[i]) or np.isnan(ema50_12h_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or opposite signal
            if close[i] < donchian_low[i] or \
               (close[i] > donchian_high[i] and volume[i] > 1.5 * avg_volume[i] and close[i] < ema50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or opposite signal
            if close[i] > donchian_high[i] or \
               (close[i] < donchian_low[i] and volume[i] > 1.5 * avg_volume[i] and close[i] > ema50_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and uptrend bias
            if close[i] > donchian_high[i] and volume_ok and close[i] > ema50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and downtrend bias
            elif close[i] < donchian_low[i] and volume_ok and close[i] < ema50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals