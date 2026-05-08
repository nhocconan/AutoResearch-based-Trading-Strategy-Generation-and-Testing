#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Bollinger Band squeeze breakout with daily volume confirmation
# Long when price breaks above upper BB(20,2) on weekly chart + daily volume > 1.5x 20-day average
# Short when price breaks below lower BB(20,2) on weekly chart + daily volume > 1.5x 20-day average
# Bollinger Band squeeze identifies low volatility periods that often precede explosive moves
# Volume confirmation ensures institutional participation in the breakout
# Weekly timeframe reduces noise and false signals, targeting 30-100 trades over 4 years
# Works in both bull and bear markets as it captures volatility expansion regardless of direction

name = "1d_WeeklyBB_Squeeze_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # Calculate Bollinger Bands(20,2) on weekly data
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = pd.Series(weekly_close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Standard deviation
    dev = bb_mult * pd.Series(weekly_close).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and lower bands
    upper = basis + dev
    lower = basis - dev
    
    # Align weekly Bollinger Bands to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    basis_aligned = align_htf_to_ltf(prices, df_1w, basis)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        upper_band = upper_aligned[i]
        lower_band = lower_aligned[i]
        vol_ma_val = volume_threshold[i]
        
        if position == 0:
            # Enter long: price breaks above upper weekly BB + volume confirmation
            if price > upper_band and vol > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower weekly BB + volume confirmation
            elif price < lower_band and vol > vol_ma_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly BB basis or volume drops
            if price < basis_aligned[i] or vol < vol_ma_val * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly BB basis or volume drops
            if price > basis_aligned[i] or vol < vol_ma_val * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals