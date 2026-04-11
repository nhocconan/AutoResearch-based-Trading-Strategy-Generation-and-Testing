#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 5:
        return signals
    
    # Calculate weekly pivot points (using weekly high/low/close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point calculation
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_range = high_1w - low_1w
    
    # Weekly support and resistance levels
    r1 = weekly_pivot + (weekly_range * 1.0 / 3.0)
    s1 = weekly_pivot - (weekly_range * 1.0 / 3.0)
    r2 = weekly_pivot + (weekly_range * 2.0 / 3.0)
    s2 = weekly_pivot - (weekly_range * 2.0 / 3.0)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: 6h volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 2.0 * vol_ma_50[i]
        
        # Trend direction based on weekly pivot
        above_pivot = price_close > weekly_pivot_aligned[i]
        below_pivot = price_close < weekly_pivot_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price above weekly pivot and breaks above R1 with volume
        if above_pivot and price_close > r1_aligned[i] and vol_confirm:
            enter_long = True
        
        # Short: Price below weekly pivot and breaks below S1 with volume
        if below_pivot and price_close < s1_aligned[i] and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to opposite side of pivot
        exit_long = price_close < weekly_pivot_aligned[i]  # Cross below pivot
        exit_short = price_close > weekly_pivot_aligned[i]  # Cross above pivot
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6h strategy using weekly pivot points for trend direction and entry signals.
# Enters long when price is above weekly pivot, breaks above R1 with volume confirmation.
# Enters short when price is below weekly pivot, breaks below S1 with volume confirmation.
# Exits when price crosses back across the weekly pivot.
# Weekly pivot provides institutional reference points that work in both bull and bear markets.
# Volume confirmation filters false breakouts. 6h timeframe reduces noise vs lower timeframes.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Position size 0.25 manages risk during volatile periods. Weekly pivot adapts to changing market regimes.