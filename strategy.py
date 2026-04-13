#!/usr/bin/env python3
"""
12h_1d_1w_TripleTimeframe_Momentum
Hypothesis: Combining momentum signals across 12h (price action), 1d (volume/volatility), and 1w (trend) creates high-probability entries. 
Uses 12h price closing above/below 1d VWAP with volume expansion, filtered by 1w ADX trend strength. 
This multi-timeframe alignment reduces false signals and captures sustained moves in both bull and bear markets.
Target: 20-40 trades/year.
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
    
    # Get daily data for VWAP and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (typical price * volume)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap = vwap_num / vwap_den
    # Handle division by zero at start
    vwap = np.where(vwap_den != 0, vwap, np.nan)
    
    # Price position relative to VWAP
    price_above_vwap = close_1d > vwap
    price_below_vwap = close_1d < vwap
    
    # Volume expansion: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1d > (vol_ma_20 * 1.5)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    
    # Directional Movement
    up_move = np.where(high_1w - np.roll(high_1w, 1) > 0, high_1w - np.roll(high_1w, 1), 0)
    down_move = np.where(np.roll(low_1w, 1) - low_1w > 0, np.roll(low_1w, 1) - low_1w, 0)
    
    # Handle first values
    up_move[0] = 0
    down_move[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    plus_dm = wilders_smooth(up_move, period)
    minus_dm = wilders_smooth(down_move, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    
    # Strong trend: ADX > 25
    strong_trend = adx > 25
    
    # Align signals to 12h timeframe
    price_above_vwap_aligned = align_htf_to_ltf(prices, df_1d, price_above_vwap.astype(float))
    price_below_vwap_aligned = align_htf_to_ltf(prices, df_1d, price_below_vwap.astype(float))
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(price_above_vwap_aligned[i]) or 
            np.isnan(price_below_vwap_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Price vs VWAP + volume expansion + strong trend
        long_entry = (price_above_vwap_aligned[i] > 0.5 and 
                      volume_expansion_aligned[i] > 0.5 and 
                      strong_trend_aligned[i] > 0.5)
        short_entry = (price_below_vwap_aligned[i] > 0.5 and 
                       volume_expansion_aligned[i] > 0.5 and 
                       strong_trend_aligned[i] > 0.5)
        
        # Exit when price returns to VWAP (mean reversion signal)
        vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
        
        exit_long = position == 1 and close[i] <= vwap_aligned[i]
        exit_short = position == -1 and close[i] >= vwap_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_TripleTimeframe_Momentum"
timeframe = "12h"
leverage = 1.0