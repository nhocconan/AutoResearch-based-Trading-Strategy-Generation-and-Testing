#!/usr/bin/env python3
"""
6h_OrderBlock_OrderFlow_Imbalance_v1
Hypothesis: Order blocks (consolidation zones) with order flow imbalance (delta > 0) and 1d trend alignment capture institutional flow. 
Works in bull/bear: in bull, longs from bullish OBs with buying pressure; in bear, shorts from bearish OBs with selling pressure. 
6H timeframe reduces noise, 1d trend filter ensures directionality, volume confirmation avoids false signals. 
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Order Block detection: look for consolidation (small range) followed by strong move
    # OB = candle with small body % of range, followed by candle breaking its high/low
    body_size = np.abs(close - open_)
    candle_range = high - low
    body_ratio = np.where(candle_range > 0, body_size / candle_range, 1.0)
    
    # Identify potential OB: low body ratio (< 0.3) = consolidation
    potential_ob = body_ratio < 0.3
    
    # Bullish OB: bearish candle (close < open) with low body, followed by bullish break above its high
    bearish_candle = close < open_
    bullish_ob = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if potential_ob[i-1] and bearish_candle[i-1]:
            if close[i] > high[i-1]:  # break above OB high
                bullish_ob[i] = True
    
    # Bearish OB: bullish candle (close > open) with low body, followed by bearish break below its low
    bullish_candle = close > open_
    bearish_ob = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if potential_ob[i-1] and bullish_candle[i-1]:
            if close[i] < low[i-1]:  # break below OB low
                bearish_ob[i] = True
    
    # Order Flow Imbalance: approximate using volume and price change
    # Simplified: if close > open and volume > avg, buying pressure; vice versa
    buying_pressure = (close > open_) & (volume > vol_avg)
    selling_pressure = (close < open_) & (volume > vol_avg)
    
    # Align all indicators
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob.astype(float))
    buying_pressure_aligned = align_htf_to_ltf(prices, df_1d, buying_pressure.astype(float))
    selling_pressure_aligned = align_htf_to_ltf(prices, df_1d, selling_pressure.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        bull_ob = bullish_ob_aligned[i] > 0.5
        bear_ob = bearish_ob_aligned[i] > 0.5
        buy_pressure = buying_pressure_aligned[i] > 0.5
        sell_pressure = selling_pressure_aligned[i] > 0.5
        
        # Determine trend
        uptrend = close_val > ema50
        downtrend = close_val < ema50
        
        if position == 0:
            # Long: bullish OB + buying pressure + uptrend
            if bull_ob and buy_pressure and uptrend:
                signals[i] = size
                position = 1
            # Short: bearish OB + selling pressure + downtrend
            elif bear_ob and sell_pressure and downtrend:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: opposite OB or trend change
            if bear_ob or (close_val < ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: opposite OB or trend change
            if bull_ob or (close_val > ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_OrderBlock_OrderFlow_Imbalance_v1"
timeframe = "6h"
leverage = 1.0