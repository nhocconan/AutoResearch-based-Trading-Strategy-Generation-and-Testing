#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter, volume confirmation, and ATR-based stoploss.
Donchian(20) breakouts capture strong moves, confirmed by 12h EMA trend and above-average volume.
Exits via ATR trailing stop or Donchian reverse breakout. Target: 25-40 trades/year per symbol.
Works in bull/bear via symmetric long/short logic and volatility-based stops.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12h EMA21 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # ATR(14) for stoploss and volatility filter
    atr = np.zeros_like(close)
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(atr[i-13:i+1]) if not np.isnan(atr[i-13]) else np.nan
    
    # Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Start after warmup period
    start_idx = max(20, 21)  # Donchian(20) + EMA21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for breakouts with trend and volume confirmation
            if price > highest_high[i] and close[i-1] <= highest_high[i-1]:  # Fresh breakout above
                if ema_12h_aligned[i] > ema_12h_aligned[i-1] and vol_filter[i]:  # Uptrend + volume
                    signals[i] = size
                    position = 1
            elif price < lowest_low[i] and close[i-1] >= lowest_low[i-1]:  # Fresh breakdown below
                if ema_12h_aligned[i] < ema_12h_aligned[i-1] and vol_filter[i]:  # Downtrend + volume
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: trail stop or reverse signal
            trail_stop = highest_high[i] - 2.0 * atr_val  # Trail by 2x ATR
            reverse_signal = price < lowest_low[i] and close[i-1] >= lowest_low[i-1]
            
            if price <= trail_stop or reverse_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short position: trail stop or reverse signal
            trail_stop = lowest_low[i] + 2.0 * atr_val  # Trail by 2x ATR
            reverse_signal = price > highest_high[i] and close[i-1] <= highest_high[i-1]
            
            if price >= trail_stop or reverse_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hTrend_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0