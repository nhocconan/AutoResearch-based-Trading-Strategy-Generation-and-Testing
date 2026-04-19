#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + price above 12h EMA34 + volume > 1.5x average
# Short when price breaks below Donchian(20) low + price below 12h EMA34 + volume > 1.5x average
# Exit when price crosses opposite Donchian band or trend reverses
# Designed to capture strong trends with volume confirmation, avoiding choppy markets
# Target: 20-35 trades/year to avoid excessive fees
name = "4h_Donchian_12hEMA_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12h volume average for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 12h EMA volume
        volume_filter = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above 12h EMA + volume
            if close[i] > highest_high[i] and close[i] > ema_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below 12h EMA + volume
            elif close[i] < lowest_low[i] and close[i] < ema_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low OR trend reverses (below 12h EMA)
            if close[i] < lowest_low[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high OR trend reverses (above 12h EMA)
            if close[i] > highest_high[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals