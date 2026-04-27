#!/usr/bin/env python3
"""
4h_Pivot_Breakout_RangeFilter_v1
Hypothesis: Daily pivot points (PP, R1, S1) act as support/resistance. Breakouts above R1 with volume in uptrend, or below S1 with volume in downtrend, capture directional moves. A range filter (ADX < 25) avoids false breakouts in sideways markets. Weekly trend alignment (price vs EMA50) ensures trend-following bias. Stops at opposite pivot level. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (PP, R1, S1) from prior day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get weekly data for trend filter (price vs EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ADX(14) for range filtering: ADX < 25 = ranging (avoid breakouts)
    # Calculate directional movement
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # positive when low decreases
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di), 0.0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    range_filter = adx < 25  # Only allow breakouts when not trending strongly (avoid false breakouts in weak trends)
    
    # Align all indicators to primary timeframe (4h)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    range_filter_aligned = align_htf_to_ltf(prices, df_1d, range_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need pivot (1), EMA34 (34), EMA50 (50), volume avg (20), ADX (14+14=28)
    start_idx = max(1, 34, 50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(range_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema34_1d_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        range_ok = range_filter_aligned[i]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d) and EMA50 (1w)
            uptrend = close_val > ema34 and close_val > ema50
            downtrend = close_val < ema34 and close_val < ema50
            
            if uptrend and vol_conf and range_ok:
                # Long bias: long when price breaks above R1 with volume and not strong trend (avoid chase)
                if close_val > r1_val:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and range_ok:
                # Short bias: short when price breaks below S1 with volume and not strong trend
                if close_val < s1_val:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: price touches S1 (opposite pivot) or weekly trend fails
            if close_val < s1_val or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: price touches R1 (opposite pivot) or weekly trend fails
            if close_val > r1_val or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Pivot_Breakout_RangeFilter_v1"
timeframe = "4h"
leverage = 1.0