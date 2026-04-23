#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal Breakout with 1d Trend Filter and Volume Confirmation
- Uses daily Williams fractals for swing high/low breakout signals (proven edge in ranging/trending markets)
- 1d EMA50 defines higher timeframe trend filter: only trade in direction of 1d trend
- Volume confirmation (> 1.8x 24-period average) filters weak signals
- Exit when price retraces to 1d EMA50 or opposite fractal level
- Designed for 12h timeframe targeting 15-30 trades/year (60-120 over 4 years)
- Williams fractals provide structure-based entries that work in both bull and bear markets by capturing swing points
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams fractals on daily timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra daily bars for confirmation (center bar + 2 right bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 24-period average (adjusted for 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 24)  # for fractals, EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above bullish fractal AND above 1d EMA50 AND volume spike
            if (close[i] > bullish_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bearish fractal AND below 1d EMA50 AND volume spike
            elif (close[i] < bearish_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price retraces to 1d EMA50 OR breaks opposite fractal level
            exit_signal = False
            
            if position == 1:
                # Exit long when price retouches EMA50 OR breaks below bearish fractal
                if (close[i] <= ema_50_1d_aligned[i] or close[i] < bearish_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when price retouches EMA50 OR breaks above bullish fractal
                if (close[i] >= ema_50_1d_aligned[i] or close[i] > bullish_aligned[i]):
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