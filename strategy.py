#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla breakout with 4h trend filter and volume confirmation
    # Long when: price breaks above H3 (4h) AND price > EMA50 (4h) AND volume > 1.5x avg volume
    # Short when: price breaks below L3 (4h) AND price < EMA50 (4h) AND volume > 1.5x avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR volume drops below average
    # Session filter: 08-20 UTC to reduce noise
    # Uses discrete sizing (0.20) targeting 60-150 trades over 4 years.
    # 4h EMA50 provides trend filter to avoid counter-trend whipsaws.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivots and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivots (using previous bar's range)
    range_4h = high_4h - low_4h
    pp_4h = (high_4h + low_4h + close_4h) / 3  # Pivot point
    h3_4h = close_4h + 1.125 * range_4h
    l3_4h = close_4h - 1.125 * range_4h
    h4_4h = close_4h + 1.5 * range_4h
    l4_4h = close_4h - 1.5 * range_4h
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(vol_threshold[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > h3_4h_aligned[i]
        short_breakout = close[i] < l3_4h_aligned[i]
        
        # Trend filter: price relative to 4h EMA50
        long_trend = close[i] > ema50_4h_aligned[i]
        short_trend = close[i] < ema50_4h_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and long_trend and vol_ok and position != 1
        short_entry = short_breakout and short_trend and vol_ok and position != -1
        
        # Exit conditions: price crosses pivot point OR volume drops below average
        exit_long = close[i] < pp_4h_aligned[i] or volume[i] < vol_ma[i]
        exit_short = close[i] > pp_4h_aligned[i] or volume[i] < vol_ma[i]
        
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

name = "1h_4h_camarilla_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0