#!/usr/bin/env python3
"""
Hypothesis: 12h Bollinger Band Squeeze + Volume Spike + 1d Supertrend Trend Filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Supertrend (ATR=10, mult=3) for trend filter (price > Supertrend = uptrend, price < Supertrend = downtrend).
- Entry: Long when Bollinger Band Width < 20th percentile (squeeze) AND price breaks above upper BB AND volume > 1.5 * 12h volume MA(50) AND price > 1d Supertrend;
         Short when Bollinger Band Width < 20th percentile (squeeze) AND price breaks below lower BB AND volume > 1.5 * 12h volume MA(50) AND price < 1d Supertrend.
- Exit: Long exits when price crosses below middle BB (20 SMA); Short exits when price crosses above middle BB.
- Signal size: 0.25 discrete to balance capture and fee control.
- Bollinger Squeeze captures low volatility contraction before expansion; volume spike confirms breakout conviction; Supertrend filters higher-timeframe trend.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 12h (20, 2)
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = basis + dev
    lower_bb = basis - dev
    bb_width = (upper_bb - lower_bb) / basis * 100  # Percentage width
    
    # Bollinger Band Width percentile (20th) for squeeze detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).rank(pct=True).values * 100
    squeeze = bb_width_percentile < 20  # Below 20th percentile
    
    # Volume confirmation: 1.5x threshold on 12h volume MA(50)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, mult=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = basic_lb[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = basic_ub[i]
            direction[i] = 1
        else:
            supertrend[i] = basic_lb[i]
            direction[i] = -1
    
    # Align Supertrend and direction to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(bb_length, 50, 10)  # BB needs 20, vol MA needs 50, Supertrend needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(basis[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(squeeze[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 1d Supertrend
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm and squeeze[i]:
                # Long: price breaks above upper Bollinger Band
                if curr_high > upper_bb[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm and squeeze[i]:
                # Short: price breaks below lower Bollinger Band
                if curr_low < lower_bb[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below middle BB (20 SMA)
            if curr_close < basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above middle BB
            if curr_close > basis[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BBand_Squeeze_VolumeSpike_1dSupertrend_Trend_v1"
timeframe = "12h"
leverage = 1.0