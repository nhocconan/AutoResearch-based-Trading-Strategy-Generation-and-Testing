#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses 1-week Donchian channels to establish long-term trend direction.
# Enters on 6-hour breakouts in the direction of weekly trend only.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Stop loss at 2x ATR(14) to manage risk.
# Works in bull/bear markets by only trading with the weekly trend.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels for trend direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    upper_1w = np.full(len(high_1w), np.nan)
    lower_1w = np.full(len(low_1w), np.nan)
    
    for i in range(19, len(high_1w)):
        upper_1w[i] = np.max(high_1w[i-19:i+1])
        lower_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly trend: 1 if close > upper, -1 if close < lower, 0 otherwise
    trend_1w = np.full(len(high_1w), 0)
    for i in range(19, len(high_1w)):
        if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i])):
            if close[i] > upper_1w[i]:
                trend_1w[i] = 1
            elif close[i] < lower_1w[i]:
                trend_1w[i] = -1
    
    # Align weekly trend to 6h timeframe
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # 6-hour Donchian channels (20-period) for entry signals
    upper_6h = np.full(n, np.nan)
    lower_6h = np.full(n, np.nan)
    
    for i in range(19, n):
        upper_6h[i] = np.max(high[i-19:i+1])
        lower_6h[i] = np.min(low[i-19:i+1])
    
    # ATR(14) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if close[i] < stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if close[i] > stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long breakout: price breaks above 6h upper Donchian with weekly uptrend
                if (close[i] > upper_6h[i] and close[i-1] <= upper_6h[i] and 
                    trend_1w_aligned[i] == 1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below 6h lower Donchian with weekly downtrend
                elif (close[i] < lower_6h[i] and close[i-1] >= lower_6h[i] and 
                      trend_1w_aligned[i] == -1):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals