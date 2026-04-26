#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, enter long when price breaks above the most recent bearish Williams fractal AND 1w trend is up (close > EMA34) AND volume > 2.0x 20-period average. Enter short when price breaks below the most recent bullish Williams fractal AND 1w trend is down (close < EMA34) AND volume spike. Uses Williams fractals for precise swing points, 1w EMA34 for higher timeframe trend alignment, and volume confirmation for institutional participation. Designed for low trade frequency (10-25/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align Williams fractals to 1d timeframe with extra delay (2 bars) for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), volume MA warmup (20), fractal calculation (need at least 5 bars)
    start_idx = max(34, 20, 5)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Williams fractals
        breakout_above_bearish = close[i] > bearish_fractal_aligned[i]
        breakout_below_bullish = close[i] < bullish_fractal_aligned[i]
        
        # 1w trend filter
        trend_uptrend = close[i] > ema_34_1w_aligned[i]
        trend_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price above bearish fractal + 1w uptrend + volume spike
            long_signal = breakout_above_bearish and trend_uptrend and volume_spike[i]
            
            # Short: price below bullish fractal + 1w downtrend + volume spike
            short_signal = breakout_below_bullish and trend_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below bullish fractal OR trend change to downtrend
            if close[i] < bullish_fractal_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above bearish fractal OR trend change to uptrend
            if close[i] > bearish_fractal_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsFractal_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0