#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm
Hypothesis: Williams Fractals identify swing highs/lows on 1d chart. Breakouts above recent bullish fractal or below bearish fractal with volume confirmation (>1.5x average) and 1d trend filter (price > EMA50 for longs, < EMA50 for shorts) capture momentum moves. Exits on opposite fractal touch or trend reversal. 6h timeframe targets 60-120 trades over 4 years (15-30/year). Works in bull markets via upside breakouts and bear markets via downside breakdowns. Fractals require 2-bar confirmation delay to avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals on 1d
    # Need high, low arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: high[2] is highest of high[0..4]
    # Bullish fractal: low[2] is lowest of low[0..4]
    # These need 2 extra bars for confirmation (the right side of the pattern)
    
    # Align all 1d indicators to 6h timeframe
    # Fractals need additional_delay_bars=2 for confirmation
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.5 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA50 (50), volume avg (30), and fractals (need 5+2 bars)
    start_idx = max(50, 30, 7)  # 5 for fractals + 2 delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_50_1d_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above bullish fractal and volume confirms
                if (close_val > bull_fractal) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below bearish fractal and volume confirms
                if (close_val < bear_fractal) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches bearish fractal (support) or trend changes to downtrend
            exit_condition = (close_val < bear_fractal) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches bullish fractal (resistance) or trend changes to uptrend
            exit_condition = (close_val > bull_fractal) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0