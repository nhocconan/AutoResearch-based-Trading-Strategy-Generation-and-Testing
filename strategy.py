#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_bollinger_breakout_v1
# Uses weekly Bollinger Bands to identify volatility extremes and breakouts.
# Long when price breaks above weekly upper band with volume confirmation.
# Short when price breaks below weekly lower band with volume confirmation.
# Uses 6h ADX > 25 to filter for strong trends, avoiding false signals in ranges.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "6h_1w_bollinger_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20-period, 2 std dev)
    close_1w = df_1w['close'].values
    bb_middle = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align to 6h timeframe (weekly bands update only after weekly bar closes)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    
    # Volume confirmation: volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter on 6h: only trade when ADX > 25 (strong trend)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus and Minus Directional Movement
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    adx_filter = adx > 25  # strong trend only
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or np.isnan(adx_filter[i]):
            signals[i] = 0.0
            continue
        
        # Require both volume and strong trend filters
        if not (vol_confirm[i] and adx_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly upper band with volume
        if close[i] > bb_upper_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly lower band with volume
        elif close[i] < bb_lower_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < bb_lower_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > bb_upper_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals