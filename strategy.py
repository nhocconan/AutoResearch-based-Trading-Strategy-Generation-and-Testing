#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with 12h EMA(50) trend filter and volume confirmation (>1.8x 30-period average)
- Williams Fractals identify significant swing points where price reversed
- Breakout above bearish fractal (sell signal) or below bullish fractal (buy signal) with trend alignment
- 12h EMA(50) ensures trades align with intermediate trend to avoid counter-trend whipsaws
- Volume confirmation validates breakout with institutional participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with 12h trend from fractal breakouts
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
    
    # Get 12h data for Williams Fractals and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Fractals on 12h timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: > 1.8x 30-period average on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 50, 30)  # Fractals, EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Fractal breakout conditions
        # Long: price breaks above bullish fractal (buy signal)
        # Short: price breaks below bearish fractal (sell signal)
        long_breakout = close[i] > bullish_fractal_aligned[i]
        short_breakout = close[i] < bearish_fractal_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above bullish fractal, uptrend, volume spike
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 1.8 * vol_ma_12h_aligned[i])
            
            # Short conditions: breakout below bearish fractal, downtrend, volume spike
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 1.8 * vol_ma_12h_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite fractal level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below bearish fractal or trend turns down
                if (close[i] < bearish_fractal_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above bullish fractal or trend turns up
                if (close[i] > bullish_fractal_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0