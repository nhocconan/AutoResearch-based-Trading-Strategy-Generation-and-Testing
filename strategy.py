#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 12h EMA trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume > 1.5x average.
Exit when Williams %R reverts to midpoint (-50) or trend reverses (price crosses 12h EMA50).
Designed for low trade frequency (~20-40/year) to capture mean reversion in trending markets while avoiding false signals in ranging conditions.
Works in both bull and bear markets by requiring trend confirmation via 12h EMA50 for mean reversion entries.
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
    
    # Load 12h data for EMA50 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 for 12h trend filter
    ema50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams %R (14-period) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Align HTF EMA50 to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(wr[i]) or np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = wr[i]
        ema50_val = ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume spike
            if (wr_val < -80 and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume spike
            elif (wr_val > -20 and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R reverts to -50 OR price breaks below 12h EMA50 (trend reversal)
                if wr_val >= -50 or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R reverts to -50 OR price breaks above 12h EMA50 (trend reversal)
                if wr_val <= -50 or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_12hEMA50_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0