#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R (14) mean reversion with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND close > 1d EMA50 (uptrend) AND volume > 1.5x average.
Short when Williams %R > -20 (overbought) AND close < 1d EMA50 (downtrend) AND volume > 1.5x average.
Exit on opposite Williams %R level or trend reversal. Uses 4h timeframe targeting 75-200 total trades over 4 years.
Williams %R identifies exhaustion points in both bull and bear markets, while 1d EMA50 filters medium-term trend.
Volume confirmation ensures breakout validity. Designed to capture reversals at extremes with controlled trade frequency.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 1d EMA50 (uptrend) AND volume spike
            if (wr < -80 and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R overbought (> -20) AND price < 1d EMA50 (downtrend) AND volume spike
            elif (wr > -20 and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 (momentum fading) OR trend reversal
                if (wr > -50 or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R < -50 (momentum fading) OR trend reversal
                if (wr < -50 or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_14_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0