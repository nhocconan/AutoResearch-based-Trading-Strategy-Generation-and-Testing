#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR trailing stop
# 1w Donchian(20) provides major structural support/resistance aligned with daily timeframe
# Volume confirmation (current 1d volume > 1.5x 20-period average) filters false breakouts
# ATR(14) trailing stop (3x ATR) manages risk and adapts to volatility
# Works in bull/bear: price reacts to weekly structure, volume confirms validity, ATR stop protects in whipsaws
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute ATR(14) for trailing stop
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0  # for long trailing stop
    lowest_low_since_entry = 0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_high_since_entry:
                highest_high_since_entry = close[i]
            
            # ATR trailing stop: exit if price drops 3*ATR from highest high
            trailing_stop = highest_high_since_entry - 3.0 * atr[i]
            if close[i] < trailing_stop:
                position = 0
                highest_high_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_low_since_entry:
                lowest_low_since_entry = close[i]
            
            # ATR trailing stop: exit if price rises 3*ATR from lowest low
            trailing_stop = lowest_low_since_entry + 3.0 * atr[i]
            if close[i] > trailing_stop:
                position = 0
                lowest_low_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation
            # Long on Donchian upper breakout, Short on Donchian lower breakout
            if volume_confirmed:
                if close[i] > upper_aligned[i]:
                    position = 1
                    highest_high_since_entry = close[i]
                    signals[i] = 0.25
                elif close[i] < lower_aligned[i]:
                    position = -1
                    lowest_low_since_entry = close[i]
                    signals[i] = -0.25
    
    return signals