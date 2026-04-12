#!/usr/bin/env python3
"""
4h_12h_TRIX_Volume_Spike_Crossover_v1
Hypothesis: 4h timeframe with TRIX (1-period rate of change of triple EMA) crossover signals confirmed by volume spikes and 12h trend filter.
TRIX > 0 indicates bullish momentum, TRIX < 0 indicates bearish momentum.
Volume spike (2x average) confirms conviction.
Only take signals aligned with 12h EMA(50) trend to avoid counter-trend trades.
Designed for moderate trade frequency (target 20-50/year) with strong edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_TRIX_Volume_Spike_Crossover_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate TRIX (1-period ROC of triple EMA)
    # EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 1-period ROC of EMA3
    trix = np.zeros_like(close)
    trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 2x average
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # TRIX crossover signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        # Trend filter: price above/below 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Entry conditions: TRIX crossover with volume and trend alignment
        long_entry = trix_cross_up and volume_spike and above_ema
        short_entry = trix_cross_down and volume_spike and below_ema
        
        # Exit conditions: opposite TRIX crossover
        long_exit = trix_cross_down
        short_exit = trix_cross_up
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals