#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d trend filter (EMA50) and volume confirmation.
Long when Williams %R < -80 (oversold) and price > EMA50 (uptrend) and volume > 1.3x average.
Short when Williams %R > -20 (overbought) and price < EMA50 (downtrend) and volume > 1.3x average.
Exit when Williams %R reverses to > -50 for long or < -50 for short, or trend weakens (price crosses EMA50).
Designed for low trade frequency (~10-25/year) to capture mean reversions in trending markets.
Works in both bull and bear markets by requiring trend confirmation (price vs EMA50) for mean reversion entries.
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
    
    # Load 1d data for Williams %R and EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = williams_r.values
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA50 (50-period) on 1d for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = williams_r_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) and price > EMA50 (uptrend) and volume confirmation
            if (wr_val < -80 and close_price > ema50_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) and price < EMA50 (downtrend) and volume confirmation
            elif (wr_val > -20 and close_price < ema50_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 (reversing from oversold) OR price < EMA50 (trend change)
                if wr_val > -50 or close_price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R < -50 (reversing from overbought) OR price > EMA50 (trend change)
                if wr_val < -50 or close_price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dEMA50_Volume_MeanReversion"
timeframe = "12h"
leverage = 1.0