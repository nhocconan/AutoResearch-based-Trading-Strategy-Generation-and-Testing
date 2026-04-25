#!/usr/bin/env python3
"""
6h_IchiCloud_TKCross_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, use Ichimoku cloud (Tenkan/Kijun cross + price vs cloud) filtered by 1d EMA50 trend and volume spikes (>1.8x 20-bar average).
Ichimoku provides dynamic support/resistance and trend direction; 1d EMA50 ensures alignment with higher timeframe trend; volume confirms breakout strength.
Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag. Works in bull markets via cloud breakouts and in bear markets via failed breaks/reversals near cloud edges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # The cloud is between senkou_a and senkou_b
    # For simplicity, we use: price > max(senkou_a, senkou_b) = above cloud (bullish)
    # price < min(senkou_a, senkou_b) = below cloud (bearish)
    # Note: In real Ichimoku, senkou spans are shifted forward 26 periods, but for signal generation
    # we use current values as proxy for cloud twist/trend strength (common simplification)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for Ichimoku (52), EMA50 (50), volume MA (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Cloud boundaries
        top_cloud = max(senkou_a[i], senkou_b[i])
        bottom_cloud = min(senkou_a[i], senkou_b[i])
        
        # TK cross: Tenkan > Kijun = bullish cross, Tenkan < Kijun = bearish cross
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above cloud + TK bullish cross + volume spike + 1d uptrend
            long_breakout = curr_close > top_cloud and tk_bullish
            # Short: price breaks below cloud + TK bearish cross + volume spike + 1d downtrend
            short_breakout = curr_close < bottom_cloud and tk_bearish
            
            # Trend filter: price must be on correct side of 1d EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and long_trend
            short_entry = short_breakout and volume_spike[i] and short_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below cloud OR TK bearish cross OR trend reverses
            if curr_close < top_cloud or not tk_bullish or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above cloud OR TK bullish cross OR trend reverses
            if curr_close > bottom_cloud or not tk_bearish or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchiCloud_TKCross_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0