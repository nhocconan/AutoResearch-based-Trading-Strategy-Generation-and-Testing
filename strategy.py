#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with volume confirmation and ATR trailing stop
# Williams %R identifies overbought/oversold conditions on 1d timeframe. Extreme readings (>80 for oversold, <-20 for overbought) 
# combined with 6h price action reversal at key levels provides mean reversion entries in both bull and bear markets.
# Volume confirmation filters false signals. ATR trailing stop manages risk.
# Target: 12-30 trades/year (50-120 over 4 years) with discrete sizing to minimize fee drag.

name = "6h_1d_williamsr_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero when high == low
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Pre-compute ATR(14) for 6h timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.0x ATR from highest
            if close[i] < highest_since_long - 2.0 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.0x ATR from lowest
            if close[i] > lowest_since_short + 2.0 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries at Williams %R extremes with volume confirmation
            # Long when oversold (< -80) and price shows bullish rejection
            # Short when overbought (> -20) and price shows bearish rejection
            if volume_confirmed:
                # Bullish rejection: close near high of bar
                bullish_rejection = close[i] > (high[i] + low[i]) / 2.0
                # Bearish rejection: close near low of bar  
                bearish_rejection = close[i] < (high[i] + low[i]) / 2.0
                
                if williams_r_aligned[i] < -80 and bullish_rejection:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.25
                elif williams_r_aligned[i] > -20 and bearish_rejection:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.25
    
    return signals