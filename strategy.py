#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
    # Long: price breaks above H3 level + volume > 1.3x 20-period 4h average + 1d close > 1d EMA20
    # Short: price breaks below L3 level + volume > 1.3x 20-period 4h average + 1d close < 1d EMA20
    # Uses discrete sizing (0.20) to minimize fee drag and ATR-based stoploss
    # Target: 15-37 trades/year to stay within 1h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 4h data for Camarilla pivots and volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    # Pivot = (H+L+C)/3
    # H3 = Pivot + 1.1*(H-L)
    # L3 = Pivot - 1.1*(H-L)
    pivot = (high_4h + low_4h + close_4h) / 3.0
    hl_range = high_4h - low_4h
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 4h volume average (20-period) for confirmation
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    atr_4h = np.zeros(n)  # Simplified ATR using 4h range
    
    # Calculate simplified ATR (4h range) for stoploss
    for i in range(n):
        idx_4h = i // 4  # 1h bars in 4h timeframe (4 bars per 4h)
        if idx_4h < len(high_4h) and idx_4h < len(low_4h):
            daily_range = high_4h[idx_4h] - low_4h[idx_4h]
            atr_4h[i] = daily_range * 0.5  # Approximate ATR as 50% of 4h range
        else:
            atr_4h[i] = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_4h_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            # Hold current position outside session
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        idx_4h = i // 4  # 1h bars in 4h timeframe (4 bars per 4h)
        if idx_4h >= len(volume_4h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_4h[idx_4h] > 1.3 * vol_avg_20_4h_aligned[i]
        
        # Trend filter: 1d close above/below EMA20
        uptrend = close_1d[idx_4h] > ema_20_1d_aligned[i] if idx_4h < len(close_1d) else False
        downtrend = close_1d[idx_4h] < ema_20_1d_aligned[i] if idx_4h < len(close_1d) else False
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 1.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 1.5 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 1.5 * atr_4h[i]
        
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

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0