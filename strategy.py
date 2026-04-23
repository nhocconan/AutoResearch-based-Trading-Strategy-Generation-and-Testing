#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R extreme reversal with 1w EMA50 trend filter and volume confirmation.
- Long: Williams %R(14) crosses above -80 (oversold reversal) + close > 1w EMA50 + volume > 1.5x 20-period average
- Short: Williams %R(14) crosses below -20 (overbought reversal) + close < 1w EMA50 + volume > 1.5x 20-period average
- Exit: Williams %R crosses below -50 (for long) or above -50 (for short) OR opposite Williams %R extreme
- Uses 1w EMA50 as trend filter to ensure alignment with weekly momentum
- Volume confirmation reduces false reversals
- Designed for both bull and bear markets: captures mean reversion in ranging markets and pullbacks in trends
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag on 1d timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) using previous bar to avoid look-ahead
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w EMA50 ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50)  # Need 14 for Williams %R, 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossover conditions (using previous bar for crossover detection)
        williams_r_prev = williams_r[i-1]
        williams_r_curr = williams_r[i]
        
        # Bullish reversal: Williams %R crosses above -80 from below
        bullish_cross = williams_r_prev <= -80 and williams_r_curr > -80
        # Bearish reversal: Williams %R crosses below -20 from above
        bearish_cross = williams_r_prev >= -20 and williams_r_curr < -20
        
        # Exit conditions: Williams %R crosses -50 midpoint
        exit_long = williams_r_prev > -50 and williams_r_curr <= -50
        exit_short = williams_r_prev < -50 and williams_r_curr >= -50
        
        # Volume spike confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R bullish reversal + price > 1w EMA50 + volume spike
            if bullish_cross and close[i] > ema_50_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R bearish reversal + price < 1w EMA50 + volume spike
            elif bearish_cross and close[i] < ema_50_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long on Williams %R crossing below -50 or bearish reversal
            if exit_long or bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Williams %R crossing above -50 or bullish reversal
            if exit_short or bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Extreme_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0