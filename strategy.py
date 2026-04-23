#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1w EMA50 trend + volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume > 1.5x average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses 6h timeframe to target ~15-25 trades/year, avoiding fee drag while capturing mean reversion in trends.
Works in both bull and bear markets by requiring trend confirmation via 1w EMA50 for entries.
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
    
    # Load 1w data for EMA50 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 for 1w trend filter
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicator to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Williams %R on 6h timeframe (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA50 (uptrend) AND volume spike
            if (wr < -80 and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA50 (downtrend) AND volume spike
            elif (wr > -20 and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (momentum fading)
                if wr > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (momentum fading)
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wEMA50_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0