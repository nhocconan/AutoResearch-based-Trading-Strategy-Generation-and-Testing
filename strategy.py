#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike (ATR ratio > 1.5) to capture strong moves aligned with medium-term trend during volatility expansion. Uses discrete sizing 0.25 to limit trades (~30/year). Volume spike ensures institutional participation, trend filter avoids counter-trend trades, and Camarilla levels provide high-probability entry/exit points. Designed to work in both bull (breakouts with trend) and bear (mean reversion at extremes during low volatility) markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(20) for Donchian and volume regime
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    #            S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low)
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R1 = prev_close + 0.275 * prev_range
    S1 = prev_close - 0.275 * prev_range
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for ATR ratio and EMA, 1 for Camarilla)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike filter
        size = fixed_size
        
        # Entry conditions: Camarilla R1/S1 breakout with volume spike AND aligned with 12h EMA50 trend
        long_entry = (close_val > r1_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < s1_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on close below S1 or trend reversal
            if close_val < s1_val or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on close above R1 or trend reversal
            if close_val > r1_val or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0