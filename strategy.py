#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 12h EMA34 trend filter and volume confirmation
# Uses 12h EMA34 for trend filter (intermediate trend) and 4h Williams Fractals for breakout signals
# Entry logic: Long when price breaks above most recent bullish Williams Fractal with volume spike and price > 12h EMA34
#              Short when price breaks below most recent bearish Williams Fractal with volume spike and price < 12h EMA34
# Exit logic: Exit when price crosses the 12h EMA34 (trend reversal) or opposite Williams Fractal level
# Williams Fractals require 2-bar confirmation delay after formation (additional_delay_bars=2)
# Works in both bull and bear markets by trading with the 12h trend
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "4h_WilliamsFractal_Breakout_12hEMA34_Volume"
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
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams Fractals on 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Williams Fractals: bearish (sell) fractal = high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    #                  bullish (buy) fractal = low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.full(len(high_4h), np.nan)
    bullish_fractal = np.full(len(low_4h), np.nan)
    
    for i in range(2, len(high_4h) - 2):
        if (high_4h[i] > high_4h[i-2] and high_4h[i] > high_4h[i-1] and 
            high_4h[i] > high_4h[i+1] and high_4h[i] > high_4h[i+2]):
            bearish_fractal[i] = high_4h[i]
        if (low_4h[i] < low_4h[i-2] and low_4h[i] < low_4h[i-1] and 
            low_4h[i] < low_4h[i+1] and low_4h[i] < low_4h[i+2]):
            bullish_fractal[i] = low_4h[i]
    
    # Align Williams Fractals to 4h timeframe with 2-bar confirmation delay
    # Williams Fractals need 2 extra bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_4h, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above most recent bullish Williams Fractal AND price > 12h EMA34 (uptrend) AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below most recent bearish Williams Fractal AND price < 12h EMA34 (downtrend) AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 12h EMA34 (trend change) OR break below most recent bearish Williams Fractal (reversal)
            if (close[i] < ema_34_12h_aligned[i] or 
                close[i] < bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 12h EMA34 (trend change) OR break above most recent bullish Williams Fractal (reversal)
            if (close[i] > ema_34_12h_aligned[i] or 
                close[i] > bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals