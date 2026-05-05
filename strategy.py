#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-bar MA
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-bar MA
# Donchian channels provide clear structure-based breakouts; 12h EMA50 filters for HTF trend alignment
# Volume confirmation ensures breakouts have institutional participation
# Works in bull (breakouts with volume in uptrend) and bear (breakdowns with volume in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 100-180 total trades over 4 years (25-45/year) to balance signal quality and fee drag

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_12h = np.full(len(close_12h), np.nan)
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian(20) channels
    if len(high) >= 20 and len(low) >= 20:
        # Donchian high: highest high over past 20 periods (including current)
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: lowest low over past 20 periods (including current)
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation on 4h (threshold: 1.5x 20-bar MA)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Donchian high AND price > 12h EMA50 AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian low AND price < 12h EMA50 AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian low (breakdown below channel)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian high (breakout above channel)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals