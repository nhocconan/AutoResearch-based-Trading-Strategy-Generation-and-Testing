#!/usr/bin/env python3
"""
4h_12h_camarilla_breakout_volume
Hypothesis: On 4h timeframe, price breaking above 12h Camarilla H3 or below L3 with 12h volume expansion and 12h ADX trend filter captures institutional breakout moves. Works in bull markets (breakouts continue) and bear markets (mean-reversion fails, so breakouts are rarer but stronger when they occur). Target: 20-50 trades/year to avoid fee drag.
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
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels from previous 12h bar's range
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + 1.1 * range_12h / 4
    camarilla_l3 = close_12h - 1.1 * range_12h / 4
    
    # 12h volume expansion: current volume > 1.5x 10-period average
    vol_ma_10 = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    volume_expansion = volume_12h > (vol_ma_10 * 1.5)
    
    # Calculate ADX (14-period) for trend strength
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    up_move = np.where(high_12h - np.roll(high_12h, 1) > 0, high_12h - np.roll(high_12h, 1), 0)
    down_move = np.where(np.roll(low_12h, 1) - low_12h > 0, np.roll(low_12h, 1) - low_12h, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
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
    
    plus_di = 100 * plus_dm / atr
    minus_di = 100 * minus_dm / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, period)
    strong_trend = adx > 20  # Moderate trend filter
    
    # Align all signals to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_12h, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of Camarilla H3/L3 with volume and trend
        long_break = close[i] > camarilla_h3_aligned[i]
        short_break = close[i] < camarilla_l3_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        
        # Exit when price returns to previous 12h close (mean reversion to equilibrium)
        prev_close_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        exit_long = position == 1 and close[i] <= prev_close_aligned[i]
        exit_short = position == -1 and close[i] >= prev_close_aligned[i]
        
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

name = "4h_12h_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0