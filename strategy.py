#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for low trade frequency (target: 12-37 trades/year)
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend entries
# Williams Fractals provide high-probability reversal points with natural look-ahead delay
# Volume spike (>2.0 * 50-period EMA) confirms participation on breakouts
# Designed for very low trade frequency with 0.25 sizing to manage drawdown
# Works in bull markets via breakout continuation and bear markets via trend-following alignment

name = "12h_Williams_Fractal_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Fractals on 1d (requires 2-bar confirmation after center)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n+1] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n+1] < low[n+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i-1] < high_1d[i] and 
            high_1d[i] > high_1d[i+1] and 
            high_1d[i+1] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i-1] > low_1d[i] and 
            low_1d[i] < low_1d[i+1] and 
            low_1d[i+1] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align Williams Fractals to 12h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 2.0 * 50-period EMA (12h * ~4.2 = 50 periods = ~21 days)
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above bullish fractal with volume spike
                if close[i] > bullish_fractal_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below bearish fractal with volume spike
                if close[i] < bearish_fractal_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1w EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below bearish fractal or price below 1w EMA50
            if close[i] < bearish_fractal_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above bullish fractal or price above 1w EMA50
            if close[i] > bullish_fractal_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals