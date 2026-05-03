#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above 1d Donchian upper channel AND close > 1w EMA50 AND volume > 2.0x 20-bar average
# Short when: price breaks below 1d Donchian lower channel AND close < 1w EMA50 AND volume > 2.0x 20-bar average
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR
#                      short exit when price > lowest_low_since_entry + 2.0 * ATR
# Uses 1d Donchian for structure (proven edge from top performers), 1w EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike confirmation (2.0x 20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate ATR(20) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_high = 0.0
    entry_low = 0.0
    
    # Start from lookback period to ensure valid Donchian channels
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous period's low
        
        # Trend and volume filters
        trend_filter_long = close[i] > ema_50_1w_aligned[i]
        trend_filter_short = close[i] < ema_50_1w_aligned[i]
        vol_filter = volume_spike[i]
        
        # Entry logic
        if position == 0:
            if long_breakout and trend_filter_long and vol_filter:
                signals[i] = 0.25  # Long 25%
                position = 1
                entry_high = high[i]
                entry_low = low[i]
            elif short_breakout and trend_filter_short and vol_filter:
                signals[i] = -0.25  # Short 25%
                position = -1
                entry_high = high[i]
                entry_low = low[i]
        
        # Exit logic (trailing stop)
        elif position == 1:  # Long position
            # Update highest high since entry
            if high[i] > entry_high:
                entry_high = high[i]
            # Check trailing stop: price < highest_high - 2.0 * ATR
            if not np.isnan(atr[i]) and close[i] < entry_high - 2.0 * atr[i]:
                signals[i] = 0.0  # Exit long
                position = 0
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < entry_low:
                entry_low = low[i]
            # Check trailing stop: price > lowest_low + 2.0 * ATR
            if not np.isnan(atr[i]) and close[i] > entry_low + 2.0 * atr[i]:
                signals[i] = 0.0  # Exit short
                position = 0
    
    return signals