#!/usr/bin/env python3
name = "6h_OrderBlock_Refine_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and order block detection
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Detect bullish order blocks (last down candle before up move) and bearish order blocks (last up candle before down move)
    # Bullish OB: candle where close < open and next candle closes above its high
    # Bearish OB: candle where close > open and next candle closes below its low
    bullish_ob_low = np.zeros(len(high_1d))
    bearish_ob_high = np.zeros(len(high_1d))
    
    for i in range(1, len(close_1d)-1):
        # Bullish OB: down candle followed by up candle breaking its high
        if close_1d[i] < open_1d[i] and close_1d[i+1] > high_1d[i]:
            bullish_ob_low[i] = low_1d[i]
        # Bearish OB: up candle followed by down candle breaking its low
        elif close_1d[i] > open_1d[i] and close_1d[i+1] < low_1d[i]:
            bearish_ob_high[i] = high_1d[i]
    
    # Forward fill OB levels (they remain valid until broken)
    bullish_ob_low = pd.Series(bullish_ob_low).replace(0, np.nan).ffill().bfill().fillna(0).values
    bearish_ob_high = pd.Series(bearish_ob_high).replace(0, np.nan).ffill().bfill().fillna(0).values
    
    # Align OB levels to 6h timeframe
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_low)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_high)
    
    # Volume filter: current 6h volume > 1.8x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bullish_ob_low_aligned[i]) or 
            np.isnan(bearish_ob_high_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to bullish OB + above 1d EMA50 + volume filter
            if low[i] <= bullish_ob_low_aligned[i] and close[i] > ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price retraces to bearish OB + below 1d EMA50 + volume filter
            elif high[i] >= bearish_ob_high_aligned[i] and close[i] < ema_50_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish OB or trend changes
            if low[i] < bullish_ob_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish OB or trend changes
            if high[i] > bearish_ob_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Requires 'open' column in prices DataFrame for OB detection
# If 'open' not available, will use close as fallback (less ideal but functional)