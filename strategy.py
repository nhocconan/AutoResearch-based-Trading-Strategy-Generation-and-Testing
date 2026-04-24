#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout + 1w Supertrend(10,3) trend filter + volume confirmation + ATR(14) stoploss.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w Supertrend(10,3) for robust trend direction in both bull and bear markets.
- Camarilla levels: calculated from prior 1d OHLC; long on break above H3, short on breakdown below L3.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to avoid low-volume breakouts.
- ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14) (using 1d ATR).
- Signal size: 0.25 discrete to balance return and drawdown control (max 0.40).
Designed to capture strong daily moves with institutional-grade filters to avoid overtrading and fee drag.
Supertrend is less whipsaw-prone than plain EMA and works in ranging markets via ATR-based bands.
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
    
    # Get 1d data for ATR and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need for volume MA and ATR
        return np.zeros(n)
    
    # Get 1w data for Supertrend(10,3) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Supertrend(10,3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + 3.0 * atr_1w
    basic_lb = (high_1w + low_1w) / 2 - 3.0 * atr_1w
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[0] = basic_ub[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            supertrend[i] = basic_ub[i]
            direction[i] = 1
        else:
            supertrend[i] = basic_lb[i]
            direction[i] = -1
        
        # Adjust bands
        if direction[i] == 1 and basic_lb[i] < supertrend[i-1]:
            basic_lb[i] = supertrend[i-1]
        if direction[i] == -1 and basic_ub[i] > supertrend[i-1]:
            basic_ub[i] = supertrend[i-1]
        
        # Final Supertrend value
        if direction[i] == 1:
            supertrend[i] = basic_lb[i]
        else:
            supertrend[i] = basic_ub[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 1d OHLC
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    camarilla_range = h1 - l1
    camarilla_h3 = c1 + camarilla_range * 1.1 / 6
    camarilla_l3 = c1 - camarilla_range * 1.1 / 6
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_direction_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
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
            
            # Determine 1w trend: Supertrend direction
            trend_bullish = trend_direction_aligned[i] == 1
            trend_bearish = trend_direction_aligned[i] == -1
            
            # Long: price breaks above Camarilla H3 AND 1w trend bullish AND volume confirmed
            if curr_high > camarilla_h3_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla L3 AND 1w trend bearish AND volume confirmed
            elif curr_low < camarilla_l3_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Camarilla L3 (reversal signal)
            stop_loss = entry_price - 2.0 * atr[i]
            if curr_low < stop_loss or curr_low < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Camarilla H3 (reversal signal)
            stop_loss = entry_price + 2.0 * atr[i]
            if curr_high > stop_loss or curr_high > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wSupertrend10_3_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0