#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
- Long when Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume > 2.0x 20-period average
- Short when Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume > 2.0x 20-period average
- Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) - mean reversion to midpoint
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume spike filter reduces false signals and churn
- Williams %R is effective in ranging/bear markets for mean reversion entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in both bull and bear: trend filter ensures we only trade with higher timeframe momentum
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
    
    # Get 1d data for EMA50 trend filter and Williams %R (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Williams %R looks back 14 periods by default
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average (increased threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA50, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        exit_long = williams_r_aligned[i] > -50  # Exit long when %R crosses above -50
        exit_short = williams_r_aligned[i] < -50  # Exit short when %R crosses below -50
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation (stricter threshold)
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Oversold + uptrend + volume confirmation
            if oversold and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + downtrend + volume confirmation
            elif overbought and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R mean reversion to midpoint (-50)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if exit_long:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50
                if exit_short:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0