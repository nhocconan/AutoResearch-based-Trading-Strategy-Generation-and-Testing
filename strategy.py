#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v1
# Hypothesis: Use Williams fractal breakouts on 4h combined with 1d EMA trend filter and volume confirmation.
# Works in bull markets (breakouts with trend) and bear markets (avoids counter-trend breakouts when 1d trend opposes).
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) by requiring multi-timeframe alignment and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams fractals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams fractals on 4h
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_4h['high'].values,
        df_4h['low'].values,
    )
    # Fractals need 2 extra bars for confirmation (Williams fractal requires 2 candles after the peak/trough)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_4h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_4h, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x average of last 12 periods (3 hours in 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below bullish fractal level or loses trend alignment
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above bearish fractal level or loses trend alignment
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price closes above bullish fractal with uptrend and volume
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_1d_aligned[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below bearish fractal with downtrend and volume
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals