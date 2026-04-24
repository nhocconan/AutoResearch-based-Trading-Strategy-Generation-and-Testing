#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Williams %R(14) crosses above -20 from below (short-term overbought reversal) in 1w bear trend OR crosses below -80 from above (short-term oversold reversal) in 1w bull trend, with volume > 2.0 * 6h volume MA(20).
- Exit: Williams %R returns to opposite extreme (-80 for longs, -20 for shorts) or 6 bar time-based exit.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Williams %R captures short-term exhaustion in counter-trend moves within larger weekly trends, volume confirms conviction, works in ranging markets via mean reversion and in trending markets via pullback entries.
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
    
    # Get 6h data for Williams %R and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(df_6h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_6h['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R from 6h to 6h timeframe (direct use with alignment for safety)
    wr_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1w EMA34 for trend
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_wr = wr_aligned[i-1] if i > 0 else -50
        
        # Volume confirmation: 2.0x threshold
        vol_confirmed = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        # Determine 1w EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: Williams %R crosses below -80 from above (oversold) in 1w bull trend
            if prev_wr >= -80 and wr_aligned[i] < -80 and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                bars_in_trade = 1
            # Short: Williams %R crosses above -20 from below (overbought) in 1w bear trend
            elif prev_wr <= -20 and wr_aligned[i] > -20 and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                bars_in_trade = 1
        elif position == 1:
            # Long position: exit on Williams %R return to -20 or time-based exit (6 bars)
            bars_in_trade += 1
            if wr_aligned[i] >= -20 or bars_in_trade >= 6:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Williams %R return to -80 or time-based exit (6 bars)
            bars_in_trade += 1
            if wr_aligned[i] <= -80 or bars_in_trade >= 6:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_extreme_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0