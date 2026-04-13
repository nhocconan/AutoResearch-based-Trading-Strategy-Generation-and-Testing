#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Long when: price breaks above 12h H3 AND 1d close > 1d EMA50 AND volume > 1.5x avg volume
    # Short when: price breaks below 12h L3 AND 1d close < 1d EMA50 AND volume > 1.5x avg volume
    # Exit when: price crosses 12h midpoint (H3/L3 average) OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1d EMA50 trend filter providing adaptive bias.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivots (using previous day's range)
    range_12h = high_12h - low_12h
    h3_12h = close_12h + 1.125 * range_12h
    l3_12h = close_12h - 1.125 * range_12h
    h4_12h = close_12h + 1.5 * range_12h
    l4_12h = close_12h - 1.5 * range_12h
    
    # Align 12h Camarilla levels to 12h timeframe
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > h3_12h_aligned[i]
        short_breakout = close[i] < l3_12h_aligned[i]
        
        # Trend filter
        long_trend = close[i] > ema_50_1d_aligned[i]
        short_trend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_trend and vol_ok and position != 1
        short_entry = short_breakout and short_trend and vol_ok and position != -1
        
        # Exit conditions: price crosses 12h midpoint OR volume drops below average
        midpoint_12h = (h3_12h_aligned[i] + l3_12h_aligned[i]) / 2
        exit_long = close[i] < midpoint_12h or volume[i] < vol_ma[i]
        exit_short = close[i] > midpoint_12h or volume[i] < vol_ma[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_ema50_volume_v1"
timeframe = "12h"
leverage = 1.0