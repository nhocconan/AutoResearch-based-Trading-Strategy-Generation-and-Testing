#!/usr/bin/env python3
# 12h_1d_williams_alligator_v1
# Strategy: 12-hour Williams Alligator crossover with 1-day trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Williams Alligator (Jaw, Teeth, Lips) crossover identifies emerging trends.
# Long when Lips > Teeth > Jaw + price above 1-day EMA50 + volume > 1.5x average.
# Short when Lips < Teeth < Jaw + price below 1-day EMA50 + volume > 1.5x average.
# Works in bull by catching uptrends early and in bear by catching downtrends early.
# Williams Alligator uses SMAs with specific periods and forward shifts to avoid look-ahead.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def williams_alligator(high, low, close):
    """Calculate Williams Alligator lines: Jaw, Teeth, Lips"""
    median_price = (high + low) / 2
    
    # Jaw: Blue line - 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    
    # Teeth: Red line - 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    
    # Lips: Green line - 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h data
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # 12h Relative Volume: current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_avg_20 + 1e-10)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Williams Alligator conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Volume confirmation: vol_ratio > 1.5
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: Alligator alignment + volume + trend alignment
        if bullish_alignment and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_alignment and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Alligator alignment with volume confirmation
        elif position == 1 and bearish_alignment and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_alignment and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals