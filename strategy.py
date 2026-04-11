#!/usr/bin/env python3
# 4h_1d_williams_alligator_v1
# Strategy: 4h Williams Alligator crossover with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phases.
# Long when Lips > Teeth > Jaw (bullish alignment) with 1d close > 1d EMA50 and volume > 1.5x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) with 1d close < 1d EMA50 and volume confirmation.
# Uses Alligator's natural filtering to reduce whipsaw, targeting 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_williams_alligator_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 4h: Jaw (13), Teeth (8), Lips (5) SMAs with future shifts
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if any required data is invalid
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Williams Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # 1d trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Bullish Alligator alignment AND bullish 1d trend AND volume confirmation
        if bullish_alignment and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish Alligator alignment AND bearish 1d trend AND volume confirmation
        elif bearish_alignment and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite Alligator alignment (trend change)
        elif position == 1 and bearish_alignment:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_alignment:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals