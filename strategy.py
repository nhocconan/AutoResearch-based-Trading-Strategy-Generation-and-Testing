#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal reversal with 1d trend filter and volume confirmation.
# Go long when bullish fractal forms (potential bottom) + price > 1d EMA50 + volume spike.
# Go short when bearish fractal forms (potential top) + price < 1d EMA50 + volume spike.
# Exit when opposite fractal forms or volume drops below average.
# Works in ranging markets (fractal reversals) and trending markets (breakouts with volume).
# Target: 25-40 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for fractals and EMA50
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: 5-bar pattern (requires 2 bars on each side)
    # Bullish: low[i-2] > low[i] and low[i-1] > low[i] and low[i+1] > low[i] and low[i+2] > low[i]
    # Bearish: high[i-2] < high[i] and high[i-1] < high[i] and high[i+1] < high[i] and high[i+2] < high[i]
    n_1d = len(high_1d)
    bullish = np.zeros(n_1d, dtype=bool)
    bearish = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (low_1d[i-2] > low_1d[i] and low_1d[i-1] > low_1d[i] and 
            low_1d[i+1] > low_1d[i] and low_1d[i+2] > low_1d[i]):
            bullish[i] = True
        if (high_1d[i-2] < high_1d[i] and high_1d[i-1] < high_1d[i] and 
            high_1d[i+1] < high_1d[i] and high_1d[i+2] < high_1d[i]):
            bearish[i] = True
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams fractals need 2-bar confirmation after the center bar
    bullish_fractal = bullish.astype(float)  # 1.0 where bullish, 0 otherwise
    bearish_fractal = bearish.astype(float)  # 1.0 where bearish, 0 otherwise
    
    # Align to 4h with 2-bar additional delay for fractal confirmation
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bullish_signal = bullish_aligned[i] > 0.5
        bearish_signal = bearish_aligned[i] > 0.5
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: bullish fractal + volume spike + price > EMA50
            if bullish_signal and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish fractal + volume spike + price < EMA50
            elif bearish_signal and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: opposite fractal forms or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bearish fractal forms or volume drops
                if bearish_signal or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bullish fractal forms or volume drops
                if bullish_signal or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Fractal_Reversal_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0