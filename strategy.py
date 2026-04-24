#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d Supertrend(10,3) trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Supertrend(10,3) for trend direction (bullish when price > Supertrend, bearish when price < Supertrend).
- Donchian(20): long on breakout above upper band, short on breakdown below lower band.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to filter weak signals.
- ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14).
- Signal size: 0.25 discrete to balance return and drawdown control.
Designed to capture strong trending moves with proper filtering to avoid overtrading and fee drag.
Supertrend is more adaptive than EMA and works well in both trending and ranging markets.
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
    
    # Get 1d data for Supertrend(10,3) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Supertrend(10,3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr_1d
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr_1d
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = basic_ub[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        # Calculate upper and lower bands
        ub = basic_ub[i]
        lb = basic_lb[i]
        
        # Update bands based on trend
        if supertrend[i-1] <= ub:
            ub = supertrend[i-1]
        if supertrend[i-1] >= lb:
            lb = supertrend[i-1]
            
        # Determine trend
        if close_1d[i] <= supertrend[i-1]:
            direction[i] = -1
            supertrend[i] = ub
        else:
            direction[i] = 1
            supertrend[i] = lb
    
    # Align HTF Supertrend and direction to 4h timeframe
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 20, 14, 20)  # Need enough bars for Supertrend, Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Determine 1d trend: bullish if direction = 1, bearish if direction = -1
            trend_bullish = direction_1d_aligned[i] == 1
            trend_bearish = direction_1d_aligned[i] == -1
            
            # Long: Donchian breakout above upper band AND 1d trend bullish AND volume confirmed
            if curr_high > highest_high[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakdown below lower band AND 1d trend bearish AND volume confirmed
            elif curr_low < lowest_low[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or Donchian break below lower band
            stop_loss = entry_price - 2.0 * atr[i]
            if curr_low < stop_loss or curr_low < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or Donchian break above upper band
            stop_loss = entry_price + 2.0 * atr[i]
            if curr_high > stop_loss or curr_high > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dSupertrend10_3_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0