#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with weekly trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter and 1d for Camarilla pivot levels.
- Camarilla pivot: calculates H3/L3 from previous 1d OHLC (fading/breakout levels).
- Entry: Long when price breaks above H3 AND price > 1w EMA34 AND volume > 1.5 * 20-period average volume (aligned).
         Short when price breaks below L3 AND price < 1w EMA34 AND volume > 1.5 * 20-period average volume (aligned).
- Exit: Opposite Camarilla breakout signal or price returns to pivot point (mean reversion).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla H3/L3 acts as breakout/continuation levels; weekly EMA34 filters counter-trend trades.
- Volume confirmation ensures breakout legitimacy in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels: returns H3, L3, and pivot point."""
    range_ = high - low
    H3 = close + (range_ * 1.1 / 4)
    L3 = close - (range_ * 1.1 / 4)
    pivot = (high + low + close) / 3
    return H3, L3, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d Camarilla pivot levels (H3, L3, pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for previous day's OHLC
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    H3, L3, pivot = camarilla_pivot(prev_high, prev_low, prev_close)
    
    # Align Camarilla levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 2, 20)  # Need 34 for 1w EMA, 2 for 1d OHLC shift, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below L3 OR price returns to pivot point (mean reversion)
            if position == 1:
                if curr_close < L3_aligned[i] or abs(curr_close - pivot_aligned[i]) < 0.001 * pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR price returns to pivot point (mean reversion)
            elif position == -1:
                if curr_close > H3_aligned[i] or abs(curr_close - pivot_aligned[i]) < 0.001 * pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend filter and volume confirmation
        if position == 0:
            # Camarilla breakout signals
            breakout_up = curr_close > H3_aligned[i] and prev_close <= H3_aligned[i]
            breakout_down = curr_close < L3_aligned[i] and prev_close >= L3_aligned[i]
            
            # Trend filter: price vs 1w EMA34
            long_trend = curr_close > ema34_1w_aligned[i]
            short_trend = curr_close < ema34_1w_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if breakout_up and long_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif breakout_down and short_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA34_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0