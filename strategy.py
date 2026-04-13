#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
    # Long: price breaks above Donchian upper AND price > 1d EMA50 AND volume > 1.5x 20-period avg
    # Short: price breaks below Donchian lower AND price < 1d EMA50 AND volume > 1.5x 20-period avg
    # Exit: price returns to Donchian midpoint (mean reversion)
    # Using 1d EMA for trend filter (structure) and 4h for entry timing
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 20-50 trades/year (~80-200 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])
        midpoint[i] = (upper[i] + lower[i]) / 2
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(midpoint[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA50, only short if price < 1d EMA50
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > upper[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < lower[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to midpoint (mean reversion)
        long_exit = close[i] < midpoint[i]
        short_exit = close[i] > midpoint[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0