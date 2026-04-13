#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
    # Long: price breaks above 20-period high + volume > 1.5x 20-period average + 1d close > 1d EMA20
    # Short: price breaks below 20-period low + volume > 1.5x 20-period average + 1d close < 1d EMA20
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-40 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 12h data for volume confirmation and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume average (20-period) for confirmation
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_12h = np.zeros(n)  # Simplified ATR using 12h range
    
    # Calculate simplified ATR (12h range) for stoploss
    for i in range(n):
        idx_12h = i // 3  # 4h bars in 12h timeframe (3 bars per 12h)
        if idx_12h < len(high_12h) and idx_12h < len(low_12h):
            range_12h = high_12h[idx_12h] - low_12h[idx_12h]
            atr_12h[i] = range_12h * 0.5  # Approximate ATR as 50% of 12h range
        else:
            atr_12h[i] = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        idx_12h = i // 3  # 4h bars in 12h timeframe (3 bars per 12h)
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: 1d close above/below EMA20
        uptrend = close_1d[idx_12h] > ema_20_1d_aligned[i] if idx_12h < len(close_1d) else False
        downtrend = close_1d[idx_12h] < ema_20_1d_aligned[i] if idx_12h < len(close_1d) else False
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > high_20_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < low_20_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 1.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 1.5 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 1.5 * atr_12h[i]
        
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

name = "4h_12h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0