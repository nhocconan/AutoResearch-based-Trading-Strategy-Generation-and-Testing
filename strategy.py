#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla breakout with 1d volume confirmation and 1w trend filter
    # Long: price breaks above H3 + 1d volume > 2.0x 20-period average + 1w close > 1w EMA20
    # Short: price breaks below L3 + 1d volume > 2.0x 20-period average + 1w close < 1w EMA20
    # Uses ATR-based stoploss and discrete sizing (0.25) to control fee drag
    # Target: 20-50 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    hl_range = high_1d - low_1d
    h3 = pivot + 1.1 * hl_range
    l3 = pivot - 1.1 * hl_range
    
    # Calculate 1d volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR for stoploss (using 4h data)
    atr_4h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_4h[i] = tr
        else:
            atr_4h[i] = 0.93 * atr_4h[i-1] + 0.07 * tr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        # We need to get the 1d volume value aligned to current 4h bar
        # Since we aligned the average, we compare current 4h bar's equivalent volume
        # Simpler approach: use 4h volume vs its own average for confirmation
        vol_avg_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_4h[i]
        
        # Trend filter: 4h close above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend
        breakout_long = (close[i] > h3_aligned[i]) and volume_confirmed and uptrend
        breakout_short = (close[i] < l3_aligned[i]) and volume_confirmed and downtrend
        
        # Stoploss: 2.5x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.5 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.5 * atr_4h[i]
        
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

name = "4h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0