#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly EMA50 trend filter and volume confirmation
# Uses weekly EMA50 for long-term trend direction (1w EMA50) and 6h Williams Fractals for breakout signals
# Entry logic: Long when price breaks above most recent 6h bullish fractal with volume spike and price > weekly EMA50 (uptrend)
#              Short when price breaks below most recent 6h bearish fractal with volume spike and price < weekly EMA50 (downtrend)
# Exit logic: Exit when price crosses the weekly EMA50 (trend reversal) or opposite fractal level
# Works in both bull and bear markets by trading with the weekly trend
# Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Williams Fractals (LTf)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Williams Fractals: bearish = high[i] is highest of 5 bars (i-2,i-1,i,i+1,i+2)
    #                 bullish = low[i] is lowest of 5 bars (i-2,i-1,i,i+1,i+2)
    bearish_fractal = np.full(len(high_6h), np.nan)
    bullish_fractal = np.full(len(low_6h), np.nan)
    
    for i in range(2, len(high_6h) - 2):
        if (high_6h[i] > high_6h[i-2] and high_6h[i] > high_6h[i-1] and 
            high_6h[i] > high_6h[i+1] and high_6h[i] > high_6h[i+2]):
            bearish_fractal[i] = high_6h[i]
        if (low_6h[i] < low_6h[i-2] and low_6h[i] < low_6h[i-1] and 
            low_6h[i] < low_6h[i+1] and low_6h[i] < low_6h[i+2]):
            bullish_fractal[i] = low_6h[i]
    
    # Align fractals to 6h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 6h bullish fractal AND price > weekly EMA50 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 6h bearish fractal AND price < weekly EMA50 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below weekly EMA50 (trend change) OR break below 6h bearish fractal (reversal)
            if (close[i] < ema_50_1w_aligned[i] or 
                close[i] < bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above weekly EMA50 (trend change) OR break above 6h bullish fractal (reversal)
            if (close[i] > ema_50_1w_aligned[i] or 
                close[i] > bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals