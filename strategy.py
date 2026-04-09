#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Williams %R with 1d EMA filter
# Williams %R(14) on weekly timeframe identifies overextended conditions
# Entry: Long when weekly %R < -80 (oversold) and price > 1d EMA50 (bullish bias)
# Entry: Short when weekly %R > -20 (overbought) and price < 1d EMA50 (bearish bias)
# Exit: Opposite %R level or EMA cross
# Volume confirmation: current 6h volume > 1.5x 20-period average to filter weak moves
# Works in bull/bear: mean reversion from weekly extremes with trend filter
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_1d_williamsr_ema_volume_v1"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Williams %R(14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w) * -100.0
    
    # Align weekly Williams %R to 6h timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # Load daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate daily EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit on weekly %R > -50 (recovery from oversold) or price < EMA50
            if williams_r_1w_aligned[i] > -50.0 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on weekly %R < -50 (decline from overbought) or price > EMA50
            if williams_r_1w_aligned[i] < -50.0 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion entries with volume and trend confirmation
            if volume_confirmed:
                # Long: weekly oversold + price above EMA50 (bullish bias)
                if williams_r_1w_aligned[i] < -80.0 and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: weekly overbought + price below EMA50 (bearish bias)
                elif williams_r_1w_aligned[i] > -20.0 and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals