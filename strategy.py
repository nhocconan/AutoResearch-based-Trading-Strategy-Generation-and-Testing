#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation.
    # Williams %R identifies overbought/oversold conditions; mean reversion works in ranging markets.
    # 12h EMA(50) filters trades to align with intermediate trend, reducing counter-trend whipsaws.
    # Volume spike (>1.8x 20-period MA) confirms reversal momentum.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 80-180 total trades over 4 years (20-45/year) within 6h limits.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R and EMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We need min_periods=14 for rolling max/min
    high_max_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    low_min_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (high_max_14 - close_12h) / (high_max_14 - low_min_14)
    # Handle division by zero (when high==low)
    williams_r = np.where((high_max_14 - low_min_14) == 0, -50, williams_r)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA for confirmation on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8 * 20-period MA
        volume_filter = volume[i] > 1.8 * volume_ma[i]
        
        # Trend filter: price above/below 12h EMA(50)
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Williams %R mean reversion conditions
        # Long when oversold (< -80) and volume spike in uptrend context
        long_signal = (williams_r_aligned[i] < -80) and volume_filter and uptrend
        # Short when overbought (> -20) and volume spike in downtrend context
        short_signal = (williams_r_aligned[i] > -20) and volume_filter and downtrend
        
        # Exit conditions: Williams %R returns to midpoint (-50) or opposite extreme
        long_exit = williams_r_aligned[i] > -50
        short_exit = williams_r_aligned[i] < -50
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_williams_r_mean_reversion_volume_v1"
timeframe = "6h"
leverage = 1.0