#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal Breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for trend filter and daily Williams fractals for structure (long-term support/resistance)
# Entry logic: Long when price breaks above daily bullish fractal with volume spike and price > 12h EMA50 (uptrend)
#              Short when price breaks below daily bearish fractal with volume spike and price < 12h EMA50 (downtrend)
# Exit logic: Exit when price crosses the 12h EMA50 (trend reversal)
# Williams fractals require 2-bar confirmation after formation (additional_delay_bars=2)
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_WilliamsFractal_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Williams fractals (require 2-bar confirmation after formation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams fractals: 5-bar pattern (high/low surrounded by 2 lower highs/2 higher lows)
    # Bullish fractal: low[i] is lowest of low[i-2:i+3]
    # Bearish fractal: high[i] is highest of high[i-2:i+3]
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bullish fractal: current low is lowest of surrounding 2 bars each side
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        
        # Bearish fractal: current high is highest of surrounding 2 bars each side
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align fractals to 4h timeframe with additional 2-bar delay for confirmation
    # Williams fractals need 2 extra completed 1d bars after formation for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above daily bullish fractal AND price > 12h EMA50 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below daily bearish fractal AND price < 12h EMA50 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 12h EMA50 (trend change)
            if close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 12h EMA50 (trend change)
            if close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals