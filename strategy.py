#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w EMA50 trend filter and volume confirmation
# Williams Fractals identify significant swing points where price reverses
# Breakouts above bearish fractals or below bullish fractals with volume confirm strong momentum
# 1w EMA50 ensures alignment with long-term trend to avoid counter-trend trades
# Volume spike filter (2.0x average) ensures institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 to balance profit potential and fee drag

name = "12h_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractals: bearish (sell) = high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    #                  bullish (buy)  = low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1w = len(high_1w)
    bearish_fractal = np.full(n_1w, np.nan)
    bullish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and
            high_1w[i-3] < high_1w[i-1] and
            high_1w[i+1] < high_1w[i-1]):
            bearish_fractal[i-1] = high_1w[i-1]  # Place at the bar where fractal completes
        
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and
            low_1w[i-3] > low_1w[i-1] and
            low_1w[i+1] > low_1w[i-1]):
            bullish_fractal[i-1] = low_1w[i-1]  # Place at the bar where fractal completes
    
    # Align 1w fractals to 12h timeframe with additional delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: Price breaks above bearish fractal AND price > 1w EMA50 AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below bullish fractal AND price < 1w EMA50 AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below bullish fractal (mean reversion) OR closes below 1w EMA50 (trend change)
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above bearish fractal (mean reversion) OR closes above 1w EMA50 (trend change)
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals