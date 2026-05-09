#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_W1_Trend_Breakout"
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
    
    # Get 1d data for Williams fractals (confirmation requires 2 extra bars)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d high/low
    # Bearish fractal: high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n-2] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n-2] and low[n] < low[n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish = np.zeros(len(high_1d), dtype=bool)
    bullish = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish[i] = True
    
    # Align fractals to 6h with 2-bar confirmation delay (Williams fractals need 2 bars after)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish.astype(float), additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish.astype(float), additional_delay_bars=2)
    
    # Weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(bearish_aligned[i]) or 
            np.isnan(bullish_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bearish_signal = bearish_aligned[i] > 0.5
        bullish_signal = bullish_aligned[i] > 0.5
        ema50_val = ema50_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Bullish fractal + above weekly EMA50 + volume spike
            if bullish_signal and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish fractal + below weekly EMA50 + volume spike
            elif bearish_signal and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish fractal appears or price below weekly EMA50
            if bearish_signal or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish fractal appears or price above weekly EMA50
            if bullish_signal or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals