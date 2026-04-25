#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Williams fractals identify significant swing highs/lows on 1d timeframe.
Breakouts above bearish fractals (resistance) or below bullish fractals (support)
with volume confirmation and aligned 1d EMA34 trend capture swing moves in both bull and bear markets.
Uses 1d timeframe for fractal structure and trend to reduce noise while maintaining alignment with 12h structure.
Designed for low trade frequency (12-37/year) with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data (requires 5 bars: 2 left, center, 2 right)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for fractals (5 bars), EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above bearish fractal (resistance) AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > bearish_fractal_aligned[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below bullish fractal (support) AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < bullish_fractal_aligned[i]) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below bullish fractal (support broken) OR price crosses below EMA (trend change)
            if (curr_close < bullish_fractal_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above bearish fractal (resistance broken) OR price crosses above EMA (trend change)
            if (curr_close > bearish_fractal_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0