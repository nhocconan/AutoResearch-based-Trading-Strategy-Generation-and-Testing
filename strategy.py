#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_WeeklyTrend_v1
Hypothesis: Williams fractals on daily chart identify swing points; breakouts in direction of weekly trend with volume confirmation capture strong moves while avoiding whipsaws. Weekly trend filter reduces false signals in chop. Designed for very low trade frequency (<10/year) to minimize fee drag and work in both bull and bear markets.
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
    
    # Williams fractals on daily chart (5-bar pattern: highest high/lowest low in center)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] and high[n-1] < high[n] and high[n+1] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] and low[n-1] > low[n] and low[n+1] and low[n+2] > low[n]
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish[i] = True
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish[i] = True
    
    # Weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 50-day average
    vol_avg_50d = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * vol_avg_50d)
    
    # Align daily fractals and weekly EMA to daily timeframe
    # Williams fractals need 2 extra days for confirmation (as per Williams)
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish, additional_delay_bars=2)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish, additional_delay_bars=2)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 50 days for EMA, 50 days for volume average, and fractal formation
    start_idx = max(50, 50) + 2  # +2 for fractal confirmation delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs weekly EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Check for fresh fractals (formed within last 2 days to avoid old signals)
            bullish_fractal = bullish_1d_aligned[i] or (i>0 and bullish_1d_aligned[i-1])
            bearish_fractal = bearish_1d_aligned[i] or (i>0 and bearish_1d_aligned[i-1])
            
            if uptrend and vol_conf and bullish_fractal:
                # Long: bullish fractal confirms support in uptrend
                signals[i] = size
                position = 1
            elif downtrend and vol_conf and bearish_fractal:
                # Short: bearish fractal confirms resistance in downtrend
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit: trend reversal or opposite fractal appears
            if close_val < ema50 or bearish_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: trend reversal or opposite fractal appears
            if close_val > ema50 or bullish_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsFractal_Breakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0