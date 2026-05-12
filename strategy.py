#!/usr/bin/env python3
name = "6h_OrderBlock_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mte_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    # 6h Order Block detection (simplified: strong candle with follow-through)
    # Bullish OB: strong bearish candle followed by bullish breakout
    # Bearish OB: strong bullish candle followed by bearish breakdown
    body_size = np.abs(close - open_)
    candle_range = high - low
    strong_candle = body_size > (0.6 * candle_range)  # at least 60% body
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        open_ = prices['open'].iloc[i]  # current bar open
        
        if position == 0:
            # Check for bullish order block: strong bearish candle followed by bullish breakout
            is_strong_bearish = (close[i-1] < open_[i-1]) and strong_candle[i-1]
            breaks_above = high[i] > high[i-1]  # break above prior candle high
            
            # Check for bearish order block: strong bullish candle followed by bearish breakdown
            is_strong_bullish = (close[i-1] > open_[i-1]) and strong_candle[i-1]
            breaks_below = low[i] < low[i-1]  # break below prior candle low
            
            # Long: bullish OB + above daily EMA34 + volume filter
            if is_strong_bearish and breaks_above and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish OB + below daily EMA34 + volume filter
            elif is_strong_bullish and breaks_below and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below prior candle low or below daily EMA34
            if low[i] < low[i-1] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above prior candle high or above daily EMA34
            if high[i] > high[i-1] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals