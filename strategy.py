#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
- Williams Fractals identify significant swing highs/lows that act as support/resistance
- Only trade breakouts in direction of 1d EMA(50) trend to avoid counter-trend whipsaws
- Volume confirmation (> 2.0x 20-period average) ensures breakout has momentum
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Fractals require 2-bar confirmation after formation, so we use additional_delay_bars=2
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals on daily timeframe
    # Bearish fractal: high[i] is highest among high[i-2], high[i-1], high[i], high[i+1], high[i+2]
    # Bullish fractal: low[i] is lowest among low[i-2], low[i-1], low[i], low[i+1], low[i+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align Williams Fractals to 12h timeframe with 2-bar extra delay for confirmation
    # Williams fractals need 2 extra daily bars after the center bar for confirmation
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above bullish fractal (support turned resistance) with volume
        # Short: price breaks below bearish fractal (resistance turned support) with volume
        price_above_fractal = close[i] > bullish_aligned[i]
        price_below_fractal = close[i] < bearish_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal, uptrend, volume spike
            long_signal = (price_above_fractal and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below bearish fractal, downtrend, volume spike
            short_signal = (price_below_fractal and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite fractal break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below bearish fractal or trend turns down
                if (price_below_fractal or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above bullish fractal or trend turns up
                if (price_above_fractal or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0