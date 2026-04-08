#!/usr/bin/env python3
"""
4h_fractal_breakout_1d_trend_volume_v1
Hypothesis: Use 1d Williams fractals to identify key support/resistance levels, 
then break these levels on 4h with volume confirmation and 1d trend filter.
Fractals provide strong horizontal levels that work in both trending and ranging markets.
Trend filter ensures we trade with the higher timeframe momentum.
Target: 20-30 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high) and bullish (low) fractals"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Williams fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1d, low_1d)
    
    # Need 2 extra bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA(25) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x average of last 15 periods (15*4h = 2.5 days)
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to bullish fractal or trend changes to down
            if close[i] <= bullish_fractal_aligned[i] or ema_1d_aligned[i] < ema_1d_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price returns to bearish fractal or trend changes to up
            if close[i] >= bearish_fractal_aligned[i] or ema_1d_aligned[i] > ema_1d_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal with volume and 1d uptrend
            if (close[i] > bearish_fractal_aligned[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[max(0, i-1)] and  # Uptrend confirmation
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal with volume and 1d downtrend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[max(0, i-1)] and  # Downtrend confirmation
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals