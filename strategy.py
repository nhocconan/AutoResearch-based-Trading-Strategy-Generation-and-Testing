#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout + volume spike + 1d EMA50 trend filter
# Long when price breaks above bullish fractal + volume spike + price > 1d EMA50
# Short when price breaks below bearish fractal + volume spike + price < 1d EMA50
# Exit when price crosses back through the fractal level or volume dries up
# Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2
# Target: 20-40 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams Fractals and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n1 = len(high_1d)
    bearish = np.zeros(n1, dtype=bool)
    bullish = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-1] > high_1d[i-3] and 
            high_1d[i-1] > high_1d[i+1]):
            bearish[i-1] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-1] < low_1d[i-3] and 
            low_1d[i-1] < low_1d[i+1]):
            bullish[i-1] = True
    
    # Convert to arrays (use high for bearish fractal level, low for bullish)
    bearish_level = np.where(bearish, high_1d, np.nan)
    bullish_level = np.where(bullish, low_1d, np.nan)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h with additional delay for fractal confirmation (2 bars)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_level, additional_delay_bars=2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_level, additional_delay_bars=2)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_aligned[i]) or 
            np.isnan(bullish_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bearish_level = bearish_aligned[i]
        bullish_level = bullish_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above bullish fractal + volume spike + price > EMA50
            if price > bullish_level and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below bearish fractal + volume spike + price < EMA50
            elif price < bearish_level and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through fractal level or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below bullish fractal or volume dries up
                if price < bullish_level or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above bearish fractal or volume dries up
                if price > bearish_level or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0