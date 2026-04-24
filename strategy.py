#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when close breaks above Donchian(20) upper band AND price > 1w EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when close breaks below Donchian(20) lower band AND price < 1w EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when close breaks below Donchian(10) lower band; Short exits when close breaks above Donchian(10) upper band.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian breakouts capture momentum; 1w EMA50 filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian(20) for entries
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Donchian(10) for exits (tighter for faster exits)
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 6h data for volume MA(20)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 50)  # Donchian(20) needs 20, volume MA needs 20, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(highest_high_10[i]) or 
            np.isnan(lowest_low_10[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 1w EMA50
        uptrend = curr_close > ema_50_aligned[i]
        downtrend = curr_close < ema_50_aligned[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: close breaks above Donchian(20) upper band
                if curr_close > highest_high_20[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: close breaks below Donchian(20) lower band
                if curr_close < lowest_low_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close breaks below Donchian(10) lower band
            if curr_close < lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close breaks above Donchian(10) upper band
            if curr_close > highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0