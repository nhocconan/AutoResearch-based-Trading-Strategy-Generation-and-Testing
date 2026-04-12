#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla H4/L4 breakout with weekly trend filter and volume confirmation
    # Uses 1w for signal direction (long-term trend), 6h for precise entry timing
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Session filter (08-20 UTC) reduces low-liquidity noise trades
    # H4/L4 levels provide strong breakout signals for 6h timeframe
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    # Only trades with the dominant weekly trend to avoid counter-trend whipsaws
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate previous 1w bar's Camarilla levels (H4, L4)
    # H4 = close_prev + 1.1 * (high_prev - low_prev)
    # L4 = close_prev - 1.1 * (high_prev - low_prev)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Get 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 6h volume for confirmation (>2.0x 20-period average)
    vol_ma_6h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_6h[i] = np.mean(volume[i-20:i])
    volume_spike_6h = volume > (2.0 * vol_ma_6h)
    
    # Align all indicators to LTF (6h)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike_6h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_h4_aligned[i]
        short_breakout = close[i] < camarilla_l4_aligned[i]
        
        # 1w trend filter
        bullish_trend = close[i] > ema34_1w_aligned[i]
        bearish_trend = close[i] < ema34_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_6h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_6h[i]
        
        # Exit logic: price returns to Camarilla pivot level (mean reversion)
        # Camarilla pivot = (high_prev + low_prev + close_prev) / 3
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
        
        # Exit when price returns to pivot level (within 0.25% tolerance)
        pivot_distance = abs(close[i] - camarilla_pivot_aligned[i]) / close[i]
        at_pivot = pivot_distance < 0.0025
        
        long_exit = at_pivot or not bullish_trend
        short_exit = at_pivot or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_camarilla_h4l4_ema34_volume_v1"
timeframe = "6h"
leverage = 1.0