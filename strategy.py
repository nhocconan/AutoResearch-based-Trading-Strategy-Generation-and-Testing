#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 Breakout with 1w EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Camarilla H4 level AND price > 1w EMA50 AND volume > 2.0 * 12h volume MA(20);
         Short when price breaks below Camarilla L4 level AND price < 1w EMA50 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when price breaks below Camarilla L4 level; Short exits when price breaks above Camarilla H4 level.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volume confirmation to avoid false breakouts.
- Uses Camarilla pivot levels from 1d for intraday support/resistance with weekly trend alignment to avoid counter-trend trades.
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
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 1w
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed day
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    H4 = np.full(len(high_1d), np.nan)
    L4 = np.full(len(high_1d), np.nan)
    
    # Calculate Camarilla H4 and L4 levels (using previous day's data)
    for i in range(1, len(high_1d)):
        range_prev = high_1d[i-1] - low_1d[i-1]
        H4[i] = close_1d[i-1] + range_prev * 1.1 / 2  # H4 level
        L4[i] = close_1d[i-1] - range_prev * 1.1 / 2  # L4 level
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 12h data for volume MA(20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    # Calculate volume MA(20) for 12h
    vol_12h = df_12h['volume'].values
    vol_ma = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align volume MA to 12h timeframe (already in 12h, but ensuring alignment)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above H4 level AND price > 1w EMA50 (uptrend)
                if curr_high > H4_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L4 level AND price < 1w EMA50 (downtrend)
                elif curr_low < L4_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below L4 level
            if curr_low < L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above H4 level
            if curr_high > H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0