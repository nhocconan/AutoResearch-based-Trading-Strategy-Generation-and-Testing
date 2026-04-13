#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation
    # Long: price breaks above upper Donchian channel (20-period high) AND volume > 1.5x 20-period average AND price > 12h EMA50
    # Short: price breaks below lower Donchian channel (20-period low) AND volume > 1.5x 20-period average AND price < 12h EMA50
    # Exit: price returns to the midpoint of the Donchian channel (mean reversion)
    # Using 12h for EMA50 trend filter (structure) and 4h only for entry timing and Donchian calculation
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 19-50 trades/year (~75-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h Donchian channel (20-period) - calculated on 4h data
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Midpoint for exit: (upper + lower) / 2
    midpoint = (upper + lower) / 2
    
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
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 12h EMA50, only short if price < 12h EMA50
        long_trend_ok = close[i] > ema_12h_aligned[i]
        short_trend_ok = close[i] < ema_12h_aligned[i]
        
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

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0