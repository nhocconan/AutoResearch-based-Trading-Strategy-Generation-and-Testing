#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel (20) breakout with volume confirmation and ATR-based stop.
# In bull markets: breakout above upper band = long. In bear markets: breakdown below lower band = short.
# Volume filter ensures breakout validity. ATR stop manages risk. Designed for low trade frequency (20-50/year)
# to minimize fee drag while adapting to trend via higher timeframe structure.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band = highest high of last 20 days
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band = lowest low of last 20 days
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC ===
        # Long: price breaks above upper Donchian band with volume confirmation
        # Short: price breaks below lower Donchian band with volume confirmation
        if vol_confirm:
            if close[i] > upper_aligned[i]:
                signals[i] = 0.30  # long 30%
            elif close[i] < lower_aligned[i]:
                signals[i] = -0.30  # short 30%
        
        # === EXIT LOGIC (via signal=0) ===
        # Stoploss: 2 * ATR from entry price
        # Track position via previous signal
        if i > warmup:
            prev_signal = signals[i-1]
            if prev_signal > 0:  # long position
                # Calculate entry price approximation (we don't track exact entry, use close as proxy)
                # Stop if price drops 2*ATR from current level (conservative close-based stop)
                if close[i] < close[i-1] - 2.0 * atr[i]:
                    signals[i] = 0.0
            elif prev_signal < 0:  # short position
                # Stop if price rises 2*ATR from current level
                if close[i] > close[i-1] + 2.0 * atr[i]:
                    signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0