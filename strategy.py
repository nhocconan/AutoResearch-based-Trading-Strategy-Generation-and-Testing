#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly SuperTrend with daily volatility breakout
# Uses weekly SuperTrend (ATR=10, multiplier=3) for trend direction
# and daily ATR breakout for entry timing. Volume > 2x 20-day average confirms.
# Weekly SuperTrend filters noise and aligns with major trend, reducing whipsaws.
# Daily ATR breakout captures momentum after volatility contraction.
# Designed for low trade frequency (<20/year) to minimize fee drag in bear markets.
# Works in both bull (trend following) and bear (avoids counter-trend trades).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for SuperTrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly SuperTrend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation
    atr_period = 10
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(atr_period, len(close_1w)):
        atr_1w[i] = np.nanmean(tr_1w[i-atr_period+1:i+1])
    
    # SuperTrend calculation
    factor = 3.0
    basic_ub = (high_1w + low_1w) / 2 + factor * atr_1w
    basic_lb = (high_1w + low_1w) / 2 - factor * atr_1w
    final_ub = np.full_like(basic_ub, np.nan)
    final_lb = np.full_like(basic_lb, np.nan)
    supertrend = np.full_like(close_1w, np.nan)
    
    for i in range(atr_period, len(close_1w)):
        if i == atr_period:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]) else final_lb[i-1]
        
        if i == atr_period:
            supertrend[i] = final_ub[i]
        else:
            supertrend[i] = final_ub[i] if (supertrend[i-1] == final_ub[i-1] and close_1w[i] > final_ub[i]) else \
                           final_lb[i] if (supertrend[i-1] == final_lb[i-1] and close_1w[i] < final_lb[i]) else \
                           supertrend[i-1]
    
    # Align SuperTrend to daily
    st_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Daily ATR for breakout
    atr_period_d = 14
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(atr_period_d, n):
        atr[i] = np.nanmean(tr[i-atr_period_d+1:i+1])
    
    # 20-day average volume
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position
    
    start_idx = max(atr_period_d, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(st_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from weekly SuperTrend
        # SuperTrend value indicates stop level; trend is based on price vs SuperTrend
        uptrend = price > st_1w_aligned[i]
        downtrend = price < st_1w_aligned[i]
        
        # Breakout conditions: price moves beyond ATR bands
        upper_break = price > close[i-1] + 2.0 * atr[i]
        lower_break = price < close[i-1] - 2.0 * atr[i]
        
        # Volume confirmation: spike > 2x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: bullish breakout above ATR band with uptrend and volume
            if uptrend and upper_break and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish breakdown below ATR band with downtrend and volume
            elif downtrend and lower_break and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price closes below SuperTrend (trailing stop)
            if price < st_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price closes above SuperTrend (trailing stop)
            if price > st_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklySuperTrend_ATRBreakout_Volume"
timeframe = "1d"
leverage = 1.0