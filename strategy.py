#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly trend filter and 1d volume confirmation
    # Long: price breaks above 20-period high + weekly close > weekly EMA50 + 1d volume > 1.5x 20-period average
    # Short: price breaks below 20-period low + weekly close < weekly EMA50 + 1d volume > 1.5x 20-period average
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for 6h timeframe (Wilder's smoothing)
    atr_6h = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    for i in range(1, n):
        if i < 14:
            atr_6h[i] = np.mean(tr[1:i+1])  # Simple average for warmup
        else:
            atr_6h[i] = 0.93 * atr_6h[i-1] + 0.07 * tr[i]  # Wilder's smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(high_max_20[i]) or
            np.isnan(low_min_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d[i] > 1.5 * vol_avg_20_1d_aligned[i] if not np.isnan(volume_1d[i]) else False
        
        # Weekly trend filter
        weekly_uptrend = close_1w[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close_1w[i] < ema_50_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = (close[i] > high_max_20[i-1]) and volume_confirmed and weekly_uptrend
        breakout_short = (close[i] < low_min_20[i-1]) and volume_confirmed and weekly_downtrend
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_6h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_6h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_1d_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0