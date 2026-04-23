#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 12h Williams Bearish Fractal AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 12h Williams Bullish Fractal AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 12h EMA34 (dynamic stop/reversal)
- Williams Fractals provide high-probability reversal points with inherent look-ahead protection
- 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
- Volume spike filters false breakouts and confirms institutional participation
- Target: 12-37 trades/year (50-150 total over 4 years) on 12h timeframe to minimize fee drag
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for Williams Fractals (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Williams Fractals (requires 5-bar window: 2 left, 2 right)
    bearish_fractal_12h, bullish_fractal_12h = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Williams Fractals need 2 extra 12h bars after center bar for confirmation
    bearish_fractal_12h_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal_12h, additional_delay_bars=2
    )
    bullish_fractal_12h_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal_12h, additional_delay_bars=2
    )
    
    # Get 12h EMA34 for exit condition
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need: 34 for 1d EMA, 34+2 for Williams Fractals, 34 for 12h EMA, 20 for volume MA
    start_idx = max(35, 36+2, 35, 21)  # 38
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bearish_fractal_12h_aligned[i]) or 
            np.isnan(bullish_fractal_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 12h Williams Fractals)
        breakout_up = close[i] > bearish_fractal_12h_aligned[i]   # Break above Bearish Fractal (resistance)
        breakout_down = close[i] < bullish_fractal_12h_aligned[i] # Break below Bullish Fractal (support)
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 12h EMA34 (dynamic stop/reversal)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below EMA34
                if close[i] < ema34_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above EMA34
                if close[i] > ema34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0