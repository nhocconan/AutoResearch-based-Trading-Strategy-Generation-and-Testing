#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (1w Camarilla) and volume confirmation
# Uses weekly Camarilla levels (R3/S3) for bias, 6h Donchian breakout for entry, and volume spike (>2.0x 20-period average) for confirmation.
# Weekly Camarilla provides structural bias from higher timeframe (1w), reducing false breakouts in ranging markets.
# Donchian(20) captures momentum breaks, while volume filter ensures participation.
# Discrete position sizing (0.25) balances return and drawdown control.
# Target: 50-150 total trades over 4 years (12-37/year) by requiring confluence of weekly bias, 6h breakout, and volume.
# Session filter (00-24 UTC) - 6h candles less session-sensitive, but kept for consistency.

name = "6h_Donchian20_Breakout_1wCamarilla_R3S3_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (00-24 UTC - always true for 6h, but kept for structure)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true, but maintains pattern
    
    # 1w data for Camarilla bias (R3/S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC for Camarilla calculation (no look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3 and S3 levels from prior week
    camarilla_range_1w = (high_1w - low_1w) * 1.1 / 4
    r3_1w = close_1w + camarilla_range_1w
    s3_1w = close_1w - camarilla_range_1w
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # 6h Donchian(20) breakout levels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 2.0x 20-period average (stricter for 6h to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 periods for Donchian + 1 week for Camarilla)
    start_idx = max(lookback, 168 // 6)  # max(20, 28) = 28 (1 week in 6h bars)
    
    for i in range(start_idx, n):
        # Skip if outside trading session (always true for 6h, but kept for consistency)
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above Donchian high with bullish weekly bias (above R3) and volume spike
            if close[i] > highest_high[i] and close[i] > r3_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian low with bearish weekly bias (below S3) and volume spike
            elif close[i] < lowest_low[i] and close[i] < s3_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below Donchian low OR weekly bias turns bearish (below S3)
            if close[i] < lowest_low[i] or close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above Donchian high OR weekly bias turns bullish (above R3)
            if close[i] > highest_high[i] or close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals