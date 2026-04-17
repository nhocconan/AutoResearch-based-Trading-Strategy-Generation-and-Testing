#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Williams %R mean reversion and volume confirmation.
Long when 1d Williams %R < -80 (oversold) with volume spike and price > 12h KAMA (trend filter).
Short when 1d Williams %R > -20 (overbought) with volume spike and price < 12h KAMA.
Williams %R identifies exhaustion points, KAMA adapts to market noise, volume confirms conviction.
Designed to work in ranging markets (mean reversion) and weak trends (pullbacks to KAMA).
Uses discrete position sizing (0.25) to minimize fee churn and maximize Sharpe.
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    period = 14
    highest_high = pd.Series(high_1d).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h KAMA for trend filter (adaptive to market noise)
    close_s = pd.Series(close)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = np.where(volatility == 0, 0, er)  # avoid division by zero
    # Smoothing constants: fastest SC=2/(2+1)=0.67, slowest SC=2/(30+1)=0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close_s.iloc[9]  # seed with first value
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Williams %R (14) + volume MA (20) + KAMA seed (10)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume and price > KAMA (bullish bias)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                close[i] > kama[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume and price < KAMA (bearish bias)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  close[i] < kama[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or price < KAMA
            if (williams_r_aligned[i] > -50 or 
                close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or price > KAMA
            if (williams_r_aligned[i] < -50 or 
                close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dWilliamsR_MeanReversion_Volume_KAMA_Trend"
timeframe = "12h"
leverage = 1.0