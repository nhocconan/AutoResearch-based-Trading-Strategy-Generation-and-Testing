#!/usr/bin/env python3
"""
1d_1w_camilla_breakout_v1
Hypothesis: On 1d timeframe, weekly Camarilla H4/L4 breakouts with volume expansion and weekly ADX trend filter capture institutional moves. Works in bull markets (breakouts continue) and bear markets (mean-reversion fails, so breakouts are rarer but stronger). Target: 10-25 trades/year to avoid fee drag.
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
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla levels from previous week's range
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    range_1w = high_1w - low_1w
    camarilla_h4 = close_1w + 1.1 * range_1w / 2
    camarilla_l4 = close_1w - 1.1 * range_1w / 2
    
    # Weekly volume expansion: current volume > 1.5x 10-period average
    vol_ma_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_expansion = volume_1w > (vol_ma_10 * 1.5)
    
    # Calculate ADX (14-period) for trend strength
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    up_move = np.where(high_1w - np.roll(high_1w, 1) > 0, high_1w - np.roll(high_1w, 1), 0)
    down_move = np.where(np.roll(low_1w, 1) - low_1w > 0, np.roll(low_1w, 1) - low_1w, 0)
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
    
    # Align all signals to daily timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1w, volume_expansion.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of Camarilla H4/L4 with volume and trend
        long_break = close[i] > camarilla_h4_aligned[i]
        short_break = close[i] < camarilla_l4_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and strong_trend_aligned[i] > 0.5
        
        # Exit when price returns to previous week's close (mean reversion to equilibrium)
        prev_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
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

name = "1d_1w_camilla_breakout_v1"
timeframe = "1d"
leverage = 1.0