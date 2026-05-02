#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend filter (long-term trend) and 12h Williams Fractals for breakout signals
# Entry logic: Long when price breaks above 12h Williams Bearish Fractal with volume spike and price > 1d EMA50
#              Short when price breaks below 12h Williams Bullish Fractal with volume spike and price < 1d EMA50
# Exit logic: Exit when price crosses the 1d EMA50 (trend reversal) or opposite Williams Fractal level
# Works in both bull and bear markets by trading with the 1d trend
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "12h_WilliamsFractal_Breakout_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Williams Fractals (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Williams Fractals: Bearish = high[i] is highest of [i-2,i-1,i,i+1,i+2]
    #                 Bullish = low[i] is lowest of [i-2,i-1,i,i+1,i+2]
    bearish_fractal = np.full(len(high_12h), np.nan)
    bullish_fractal = np.full(len(low_12h), np.nan)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i] >= high_12h[i-2] and high_12h[i] >= high_12h[i-1] and 
            high_12h[i] >= high_12h[i+1] and high_12h[i] >= high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
        if (low_12h[i] <= low_12h[i-2] and low_12h[i] <= low_12h[i-1] and 
            low_12h[i] <= low_12h[i+1] and low_12h[i] <= low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
    
    # Williams Fractals need 2 extra 12h bars for confirmation (center bar + 2 future bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 12h Williams Bearish Fractal AND price > 1d EMA50 (uptrend) AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 12h Williams Bullish Fractal AND price < 1d EMA50 (downtrend) AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA50 (trend change) OR break below 12h Williams Bullish Fractal (reversal)
            if (close[i] < ema_50_1d_aligned[i] or 
                close[i] < bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA50 (trend change) OR break above 12h Williams Bearish Fractal (reversal)
            if (close[i] > ema_50_1d_aligned[i] or 
                close[i] > bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals