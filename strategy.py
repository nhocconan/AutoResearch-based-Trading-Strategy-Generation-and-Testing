#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Williams fractals identify swing highs/lows on weekly timeframe. 
Breakouts above recent weekly bullish fractal or below bearish fractal with 
volume confirmation and aligned weekly EMA34 trend capture swing moves in both 
bull and bear markets. Uses 1w timeframe for structure to reduce noise and 
avoid overtrading, targeting 12-37 trades/year.
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
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams fractals and EMA34 trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1w data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values, df_1w['low'].values
    )
    # Williams fractals need 2 extra 1w bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for fractals, EMA, volume MA
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above latest bullish fractal AND volume spike AND price > 1w EMA34 (uptrend)
            long_entry = (curr_close > bullish_fractal_aligned[i]) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below latest bearish fractal AND volume spike AND price < 1w EMA34 (downtrend)
            short_entry = (curr_close < bearish_fractal_aligned[i]) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below bearish fractal (swing low broken) OR price crosses below EMA (trend change)
            if (curr_close < bearish_fractal_aligned[i]) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above bullish fractal (swing high broken) OR price crosses above EMA (trend change)
            if (curr_close > bullish_fractal_aligned[i]) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Fractal_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0