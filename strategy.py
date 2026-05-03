#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 1d EMA34 ensures alignment with daily trend to avoid fighting the higher timeframe bias.
# Volume spike confirms institutional participation at reversal points.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate 1d EMA34 trend filter (daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Overbought: > -20, Oversold: < -80
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 50-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        wr = williams_r[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            continue
            
        # Entry conditions
        # Long: Williams %R oversold (< -80) with volume spike and price above 1d EMA34 (uptrend)
        long_entry = (wr < -80) and vol_spike and (close[i] > ema_trend)
        # Short: Williams %R overbought (> -20) with volume spike and price below 1d EMA34 (downtrend)
        short_entry = (wr > -20) and vol_spike and (close[i] < ema_trend)
        
        # Exit conditions: mean reversion complete when Williams %R returns to neutral range (-50)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            long_exit = wr > -50  # Exit when momentum returns to neutral
        elif position == -1:  # Short position
            short_exit = wr < -50  # Exit when momentum returns to neutral
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals