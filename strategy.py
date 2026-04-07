#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian Breakout + 1d RSI Filter + Volume Confirmation
# Hypothesis: Breakouts above 6h Donchian(20) combined with daily RSI(14) > 60 for long
# or < 40 for short, and volume > 20-period average capture momentum with filter.
# Works in bull via breakouts, in bear via breakdowns, and avoids false signals in range.
# Target: 20-40 trades/year to minimize fee drag.
name = "6h_donchian_breakout_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    daily_rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 6h timeframe
    daily_rsi_6h = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(daily_rsi_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or RSI < 40
            if close[i] < low_20[i] or daily_rsi_6h[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or RSI > 60
            if close[i] > high_20[i] or daily_rsi_6h[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian high with volume and RSI > 60
            if close[i] > high_20[i] and vol_confirm and daily_rsi_6h[i] > 60:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and RSI < 40
            elif close[i] < low_20[i] and vol_confirm and daily_rsi_6h[i] < 40:
                position = -1
                signals[i] = -0.25
    
    return signals