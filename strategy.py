#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
# In 6h timeframe, we look for extreme readings (< -90 for long, > -10 for short) to catch reversals.
# Entry requires alignment with 1d EMA50 trend (long in uptrend, short in downtrend) and volume spike (>2x 20-bar avg).
# Exit when Williams %R returns to neutral zone (-50) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull/bear markets by requiring trend alignment and avoiding choppy conditions.

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50) and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    lookback = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = (highest_high - close_1d) / hl_range * -100
    
    # Align Williams %R to 6h (it changes only when 1d bar closes)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, lookback)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter
        ema_trend_up = close[i] > ema_50_1d_aligned[i]
        ema_trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        wr_oversold = wr < -90  # Extreme oversold
        wr_overbought = wr > -10  # Extreme overbought
        wr_neutral = abs(wr + 50) < 10  # Near -50 (neutral)
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Extreme oversold, 1d EMA50 uptrend, volume confirm
            if wr_oversold and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Extreme overbought, 1d EMA50 downtrend, volume confirm
            elif wr_overbought and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R returns to neutral or gets overbought
            if wr_neutral or wr_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R returns to neutral or gets oversold
            if wr_neutral or wr_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals