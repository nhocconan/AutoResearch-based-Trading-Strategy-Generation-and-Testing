#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d volume/volatility confirmation
# - Uses 1d Williams Fractals (bearish/bullish) to identify key swing points
# - Break above recent bullish fractal or below recent bearish fractal on 4h
# - Confirmed by 1d volume > 1.8x its 20-period average and 1d ATR > 1.3x its 50-period average
# - ATR(14) trailing stop: exits when price retraces 2.5x ATR from extreme
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
# - Works in bull markets (fractal breakouts continue) and bear markets (breakdowns continue)
# - Williams Fractals provide objective swing high/low levels with built-in confirmation delay
# - Stricter volume/volatility filters reduce trade frequency while maintaining edge

name = "4h_1d_williams_fractal_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # 1d ATR(14) for volatility
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR(50) average for volatility regime filter
    atr_50_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # 1d Volume > 1.8x 20-period average (balanced for trade frequency)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * avg_volume_20)
    
    # Williams Fractals on 1d (requires 5 bars: 2 left, 2 right)
    # Bearish fractal: high[2] is highest of [high[0], high[1], high[2], high[3], high[4]]
    # Bullish fractal: low[2] is lowest of [low[0], low[1], low[2], low[3], low[4]]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align 1d indicators to 4h with extra delay for fractals (need 2 bars for confirmation)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_50_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_50_avg)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_50_avg_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            atr_1d_aligned[i] <= 0 or atr_50_avg_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.5x ATR from high
            if low[i] <= highest_since_entry - (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.5x ATR from low
            if high[i] >= lowest_since_entry + (2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for fractal breakouts with volume and volatility confirmation
            # Bullish breakout: price > recent bullish fractal
            if (high[i] > bullish_fractal_aligned[i] and    # Break above bullish fractal
                volume_spike_1d_aligned[i] and             # Volume confirmation
                atr_1d_aligned[i] > (1.3 * atr_50_avg_aligned[i])):  # High volatility regime
                position = 1
                entry_price = high[i]
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Bearish breakout: price < recent bearish fractal
            elif (low[i] < bearish_fractal_aligned[i] and   # Break below bearish fractal
                  volume_spike_1d_aligned[i] and            # Volume confirmation
                  atr_1d_aligned[i] > (1.3 * atr_50_avg_aligned[i])):  # High volatility regime
                position = -1
                entry_price = low[i]
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals