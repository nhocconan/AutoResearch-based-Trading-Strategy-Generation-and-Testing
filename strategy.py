#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: At 12h timeframe, Camarilla H3/L3 breakouts with 1d EMA34 trend filter and volume spike capture swing moves in both bull and bear markets. Lower trade frequency (~15-30/year) minimizes fee drag while allowing participation in major trends. H3/L3 levels provide tighter stops than R3/S3, improving risk/reward.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA34 trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels: H3 = Close + 1.1*(High-Low)*1.1/4, L3 = Close - 1.1*(High-Low)*1.1/4
    camarilla_range = 1.1 * (prev_high - prev_low)
    h3 = prev_close + camarilla_range * 0.275  # 1.1 * 1.1 / 4 = 0.3025 -> wait, 1.1 * (High-Low) * 1.1 / 4
    # Actually: H3 = Close + 1.1*(High-Low)*1.1/4 = Close + 1.21*(High-Low)/4 = Close + 0.3025*(High-Low)
    # But standard Camarilla: H3 = Close + 1.1*(High-Low)*1.1/4
    # Let's use the correct formula: H3 = Close + (1.1 * (High-Low) * 1.1) / 4
    h3 = prev_close + (1.1 * (prev_high - prev_low) * 1.1) / 4
    l3 = prev_close - (1.1 * (prev_high - prev_low) * 1.1) / 4
    
    # Align Camarilla levels to 12h timeframe (already completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + volume MA (20) + Camarilla (2)
    start_idx = max(34, 20, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + 1d EMA34 trend alignment
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and (curr_close > ema_34_1d_aligned[i])
            short_entry = short_breakout and volume_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below H3 (failed breakout) or trend turns bearish
            if curr_close < h3_aligned[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L3 (failed breakout) or trend turns bullish
            if curr_close > l3_aligned[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0