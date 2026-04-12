#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v3
# Buys when price breaks above daily H3 with volume > 1.5x 20-period average and ADX > 20.
# Shorts when price breaks below daily L3 with same conditions.
# Exits on opposite breakout or when ADX drops below 15 (trend weakness).
# Uses discrete position sizes (0.25) to minimize fee churn.
# Designed for 15-30 trades/year to avoid fee drag while capturing breakout momentum.
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).

name = "4h_1d_camarilla_breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 4h timeframe (daily levels update only after daily bar closes)
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: enter when ADX > 20, exit when ADX < 15 (hysteresis)
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Check if we need to exit due to weak trend
        if position == 1 and adx[i] < 15:
            position = 0
            signals[i] = 0.0
            continue
        elif position == -1 and adx[i] < 15:
            position = 0
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above daily H3 with volume and ADX > 20
        if close[i] > h3_level[i] and adx[i] > 20 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below daily L3 with volume and ADX > 20
        elif close[i] < l3_level[i] and adx[i] > 20 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
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