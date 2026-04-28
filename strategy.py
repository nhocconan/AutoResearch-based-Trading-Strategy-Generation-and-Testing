#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions (-20 to -80 range).
# Extreme readings below -80 (oversold) with 1d EMA50 uptrend and volume spike = long.
# Extreme readings above -20 (overbought) with 1d EMA50 downtrend and volume spike = short.
# Exit when Williams %R returns to -50 (midpoint) or opposite extreme.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Works in both bull/bear markets by requiring alignment with 1d trend.
# Volume confirmation filters weak mean-reversion signals.

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
    
    # Get 1d data for trend filter and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling max/min for 1d
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    williams_r = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Align Williams %R to 6h (changes only when 1d bar closes)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 14)  # Ensure sufficient history for volume MA, EMA, and Williams %R
    
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
        williams_r_val = williams_r_aligned[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), 1d EMA50 uptrend, volume confirm
            if williams_r_val < -80 and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), 1d EMA50 downtrend, volume confirm
            elif williams_r_val > -20 and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R returns to -50 or goes above -20
            if williams_r_val >= -50 or williams_r_val > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R returns to -50 or goes below -80
            if williams_r_val <= -50 or williams_r_val < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals