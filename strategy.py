#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams fractals identify key swing points. Breakouts above bearish fractals or below bullish fractals with 1d EMA34 trend alignment and volume confirmation capture strong momentum moves in both bull and bear markets. Uses weekly context to filter counter-trend breakouts. Target: 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for context filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend context (lagging indicator, needs extra delay)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w, additional_delay_bars=1)
    
    # Load daily data ONCE before loop for EMA34 trend and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams fractals on daily (needs 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 34, 50) + 2
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filters
        weekly_bullish = curr_close > ema_1w_aligned[i]
        weekly_bearish = curr_close < ema_1w_aligned[i]
        daily_bullish = curr_close > ema_1d_aligned[i]
        daily_bearish = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: fractal breakout + daily trend + volume spike
            # Long: price breaks above bearish fractal AND daily bullish bias AND volume spike
            long_entry = (curr_high > bearish_fractal_aligned[i]) and daily_bullish and vol_spike
            # Short: price breaks below bullish fractal AND daily bearish bias AND volume spike
            short_entry = (curr_low < bullish_fractal_aligned[i]) and daily_bearish and vol_spike
            
            # Weekly context filter: only take longs in weekly uptrend, shorts in weekly downtrend
            if long_entry and weekly_bullish:
                signals[i] = 0.25
                position = 1
            elif short_entry and weekly_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below bullish fractal (support) OR loss of daily bullish bias
            if (curr_low < bullish_fractal_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above bearish fractal (resistance) OR loss of daily bearish bias
            if (curr_high > bearish_fractal_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0