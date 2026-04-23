#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 2.0x average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 6h timeframe with Williams %R to capture mean reversion in both bull and bear markets.
1d EMA50 provides trend filter to avoid counter-trend trades. Volume spike ensures conviction.
Target: 80-120 trades over 4 years (20-30/year) to stay within proven working range for 6h.
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
    
    # Calculate Williams %R (14-period) on 6h - ONCE before loop
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_6h[i]
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 (uptrend) AND volume spike
            if (wr < -80 and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 (downtrend) AND volume spike
            elif (wr > -20 and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (mean reversion complete)
                if wr > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (mean reversion complete)
                if wr < -50:
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