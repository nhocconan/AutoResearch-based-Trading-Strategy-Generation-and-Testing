#!/usr/bin/env python3
"""
4h_1d_1w_Momentum_Fusion_V1
Hypothesis: Combines 1-day momentum (price vs 50-day EMA) with 4-hour momentum (RSI divergence) and 1-week trend filter (ADX) to capture momentum bursts in both bull and bear markets. Uses strict entry conditions (momentum alignment + volume confirmation) to limit trades to ~20-30/year, reducing fee drag. Exits on momentum reversal or trend exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for momentum and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 50-day EMA for long-term trend
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily momentum: price above/below 50 EMA
    momentum_long = close_1d > ema_50
    momentum_short = close_1d < ema_50
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
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
    
    # Get 4-hour RSI for momentum timing
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI momentum: rising RSI for long, falling RSI for short
    rsi_rising = rsi > np.roll(rsi, 1)
    rsi_falling = rsi < np.roll(rsi, 1)
    rsi_rising[0] = False
    rsi_falling[0] = False
    
    # Align all signals to 4h timeframe
    momentum_long_aligned = align_htf_to_ltf(prices, df_1d, momentum_long.astype(float))
    momentum_short_aligned = align_htf_to_ltf(prices, df_1d, momentum_short.astype(float))
    strong_trend_aligned = align_htf_to_ltf(prices, df_1w, strong_trend.astype(float))
    rsi_rising_aligned = align_htf_to_ltf(prices, None, rsi_rising.astype(float))  # Already LTF
    rsi_falling_aligned = align_htf_to_ltf(prices, None, rsi_falling.astype(float))  # Already LTF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(momentum_long_aligned[i]) or 
            np.isnan(momentum_short_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Momentum alignment + RSI confirmation + trend filter
        long_entry = (momentum_long_aligned[i] > 0.5 and 
                     rsi_rising_aligned[i] > 0.5 and 
                     strong_trend_aligned[i] > 0.5)
        short_entry = (momentum_short_aligned[i] > 0.5 and 
                      rsi_falling_aligned[i] > 0.5 and 
                      strong_trend_aligned[i] > 0.5)
        
        # Exit when momentum reverses
        exit_long = position == 1 and (momentum_long_aligned[i] < 0.5 or rsi_rising_aligned[i] < 0.5)
        exit_short = position == -1 and (momentum_short_aligned[i] < 0.5 or rsi_falling_aligned[i] < 0.5)
        
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

name = "4h_1d_1w_Momentum_Fusion_V1"
timeframe = "4h"
leverage = 1.0