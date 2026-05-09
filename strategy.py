#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above upper Donchian band (20) and price > 12h EMA50 with volume > 1.5x 20-period EMA.
# Short when price breaks below lower Donchian band (20) and price < 12h EMA50 with volume > 1.5x 20-period EMA.
# Exit when price crosses back below/above the Donchian middle (SMA20) or volume drops.
# Uses tight entry conditions to limit trades to ~20-50/year, avoiding fee drag.
# Works in both bull and bear markets by following 12h EMA trend direction.
name = "4h_DonchianBreakout_12hEMA50_Volume"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (highest_high + lowest_low) / 2  # SMA20 equivalent
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + trend up + volume
            if price > highest_high[i] and price > ema_12h_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + trend down + volume
            elif price < lowest_low[i] and price < ema_12h_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle (SMA20) or volume drops
            if price < middle[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle (SMA20) or volume drops
            if price > middle[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals