#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: On 4h timeframe, trade Camarilla pivot R1/S1 breakouts with volume confirmation (>1.5x 20-bar average) and 12h EMA50 trend filter.
Camarilla pivots provide mathematically derived support/resistance levels that work well in ranging and trending markets.
The 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
Volume spike confirms breakout strength and reduces false signals.
Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag in BTC/ETH markets.
Works in bull markets via breakout continuations and in bear markets via mean reversion from extreme levels.
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
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter (loaded ONCE)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    # Calculate typical Camarilla levels from previous day
    # Using previous day's close as pivot for simplicity (standard Camarilla uses (H+L+C)/3)
    # But for intraday breakouts, we use previous day's range
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla R1 and S1 levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA (20) and HTF data alignment
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 + volume spike + 12h uptrend
            long_breakout = curr_close > r1_aligned[i]
            # Short: price breaks below S1 + volume spike + 12h downtrend
            short_breakout = curr_close < s1_aligned[i]
            
            # Trend filter: price must be on correct side of 12h EMA50
            long_trend = curr_close > ema_50_12h_aligned[i]
            short_trend = curr_close < ema_50_12h_aligned[i]
            
            long_entry = long_breakout and volume_spike[i] and long_trend
            short_entry = short_breakout and volume_spike[i] and short_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R1 OR trend reverses
            if curr_close < r1_aligned[i] or curr_close < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above S1 OR trend reverses
            if curr_close > s1_aligned[i] or curr_close > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0