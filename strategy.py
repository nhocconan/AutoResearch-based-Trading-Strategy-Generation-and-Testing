#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Williams fractals identify significant swing points; breakouts above/below confirm momentum
# 1w EMA50 ensures alignment with major trend to avoid counter-trend trades
# Volume confirmation filters low-strength moves
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Discrete sizing 0.25 balances profit and fee drag

name = "1d_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Williams fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractal: 5-bar pattern (t-2, t-1, t, t+1, t+2)
    # Bearish fractal: high[t] > high[t-1] and high[t] > high[t-2] and high[t] > high[t+1] and high[t] > high[t+2]
    # Bullish fractal: low[t] < low[t-1] and low[t] < low[t-2] and low[t] < low[t+1] and low[t] < low[t+2]
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and 
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and 
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align 1w fractals to 1d timeframe with 2-bar extra delay for confirmation
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_aligned[i]) or np.isnan(bullish_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price closes above bullish fractal AND price > 1w EMA50 AND volume spike
            if (close[i] > bullish_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price closes below bearish fractal AND price < 1w EMA50 AND volume spike
            elif (close[i] < bearish_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below bullish fractal (failed breakout) OR closes below 1w EMA50 (trend change)
            if close[i] < bullish_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above bearish fractal (failed breakdown) OR closes above 1w EMA50 (trend change)
            if close[i] > bearish_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals