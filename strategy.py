#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla pivot breakouts with 1d volume confirmation
    # Uses 4h for signal direction (structure), 1d for volume filter (institutional participation)
    # 1h only for precise entry timing on breakouts
    # Session filter (08-20 UTC) to avoid low-liquidity periods
    # Target: 15-37 trades/year to stay within 1h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    # Pivot = (H+L+C)/3
    # H3 = Pivot + 1.1*(H-L)
    # L3 = Pivot - 1.1*(H-L)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    hl_range_4h = high_4h - low_4h
    h3_4h = pivot_4h + 1.1 * hl_range_4h
    l3_4h = pivot_4h - 1.1 * hl_range_4h
    
    # Get 1d data for volume confirmation (institutional participation filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size (discrete level to reduce churn)
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR for 1h timeframe
    atr_1h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_1h[i] = tr  # Simple average for warmup
        else:
            atr_1h[i] = 0.93 * atr_1h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(30, n):
        # Skip if not in trading session
        if not in_session[i]:
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
        
        # Skip if data not ready
        if (np.isnan(h3_4h_aligned[i]) or 
            np.isnan(l3_4h_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = False
        # Find the most recent completed 1d bar
        # We use the aligned value which represents the last completed 1d bar
        if not np.isnan(vol_avg_20_1d_aligned[i]):
            # For volume confirmation, we compare current 1h volume to the 1d average
            # This is a proxy for institutional participation
            volume_confirmed = volume[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume confirmation
        breakout_long = (close[i] > h3_4h_aligned[i]) and volume_confirmed
        breakout_short = (close[i] < l3_4h_aligned[i]) and volume_confirmed
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_1h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_1h[i]
        
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

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0