#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND close > 1w EMA50 AND volume > 1.8x 24-period average.
Short when price breaks below 20-period Donchian low AND close < 1w EMA50 AND volume > 1.8x 24-period average.
Exit when price crosses the midpoint of the Donchian channel.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
1w EMA50 provides strong trend filter with minimal lag for 12h timeframe. Volume confirmation at 1.8x ensures institutional participation.
Designed to capture medium-term trends in both bull and bear markets by using weekly trend filter and volatility-adjusted entries.
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
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) levels from primary timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align HTF indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (24-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 1w EMA50 AND volume spike
            if (price > donchian_high[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND close < 1w EMA50 AND volume spike
            elif (price < donchian_low[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian midpoint
            if position == 1 and price < donchian_mid[i]:
                exit_signal = True
            elif position == -1 and price > donchian_mid[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0