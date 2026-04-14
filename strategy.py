#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with 4h EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold reversal) on 1d timeframe, 4h EMA > 50-period EMA (uptrend), and volume > 1.3x average.
# Short when Williams %R crosses below -80 (overbought reversal) on 1d timeframe, 4h EMA < 50-period EMA (downtrend), and volume > 1.3x average.
# Exit when Williams %R returns to -50 (midline) or EMA crossover reverses.
# Williams %R identifies overextended conditions ripe for reversal, EMA filter ensures trading with the intermediate trend,
# volume confirmation adds conviction. Designed for mean-reversion within trending environments across market cycles.
# Target: 20-35 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 4h data for EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA (9 and 50)
    ema9 = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe (primary)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema9_aligned = align_htf_to_ltf(prices, df_4h, ema9)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50)
    
    # Volume confirmation: 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(14, 50, 20)  # Need Williams %R, EMA50, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema9_aligned[i]) or
            np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # EMA trend filter
        uptrend = ema9_aligned[i] > ema50_aligned[i]
        downtrend = ema9_aligned[i] < ema50_aligned[i]
        
        # Williams %R signals: cross above -20 (long) or below -80 (short)
        # Use previous value to detect cross
        prev_williams = williams_r_aligned[i-1] if i > 0 else -50
        curr_williams = williams_r_aligned[i]
        
        cross_above_20 = prev_williams <= -20 and curr_williams > -20
        cross_below_80 = prev_williams >= -80 and curr_williams < -80
        
        # Exit when Williams %R returns to -50 or EMA crossover reverses
        cross_midline = (prev_williams > -50 and curr_williams <= -50) or (prev_williams < -50 and curr_williams >= -50)
        ema_cross_down = uptrend and ema9_aligned[i] < ema50_aligned[i]
        ema_cross_up = downtrend and ema9_aligned[i] > ema50_aligned[i]
        
        if position == 0:
            # Look for reversals in alignment with trend
            # Long: Williams %R crosses above -20 AND uptrend AND volume confirmation
            if cross_above_20 and uptrend and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -80 AND downtrend AND volume confirmation
            elif cross_below_80 and downtrend and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 or EMA crossover down
            if cross_midline or ema_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 or EMA crossover up
            if cross_midline or ema_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_EMA_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0