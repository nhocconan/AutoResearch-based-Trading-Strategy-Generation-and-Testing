#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.8x 20-day average.
Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.8x 20-day average.
Exit when price crosses 10-day EMA in opposite direction.
Uses 1d primary timeframe with 1w HTF for trend alignment to capture major moves while minimizing trades.
Designed for low frequency (target: 20-60 trades over 4 years) to reduce fee drag in bear markets.
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
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 1d (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(ema10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        donch_high = high_ma[i]
        donch_low = low_ma[i]
        ema10_val = ema10[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1w EMA50 AND volume spike
            if (price > donch_high and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low AND below 1w EMA50 AND volume spike
            elif (price < donch_low and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 10-day EMA
                if price < ema10_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 10-day EMA
                if price > ema10_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0