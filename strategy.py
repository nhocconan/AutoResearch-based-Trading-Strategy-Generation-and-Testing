#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND close > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when Williams %R > -20 (overbought) AND close < 1d EMA50 (downtrend) AND volume > 2.0x average.
Uses 6h timeframe to target 50-150 total trades over 4 years. Williams %R identifies extreme reversals,
1d EMA50 ensures alignment with higher timeframe trend, volume spike confirms conviction.
Works in both bull and bear markets by combining mean reversion with trend filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (williams_r_val < -80.0 and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (williams_r_val > -20.0 and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to neutral (> -50) OR price breaks below 1d EMA50 (trend reversal)
                if williams_r_val > -50.0 or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to neutral (< -50) OR price breaks above 1d EMA50 (trend reversal)
                if williams_r_val < -50.0 or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0