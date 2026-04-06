#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ATR Breakout with Volume Confirmation and Trend Filter.
# Uses ATR breakout from 20-period high/low with volume confirmation (1.5x average).
# Trend filter: price above/below 50-period EMA to avoid counter-trend trades.
# Works in both bull and bear markets by capturing momentum bursts.
# Target: 100-200 trades over 4 years (25-50/year).

name = "4h_atr_breakout_vol_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR calculation (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(19, n):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    # 50-period EMA for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(atr[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below 20-period low or stoploss
            stop_loss_level = entry_price - 2.5 * atr[i]
            
            if (close[i] < low_20[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above 20-period high or stoploss
            stop_loss_level = entry_price + 2.5 * atr[i]
            
            if (close[i] > high_20[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout: price closes above 20-period high AND above EMA50
                if (close[i] > high_20[i] and close[i] > ema_50[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price closes below 20-period low AND below EMA50
                elif (close[i] < low_20[i] and close[i] < ema_50[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals