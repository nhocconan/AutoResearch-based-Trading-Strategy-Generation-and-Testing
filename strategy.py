#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when: Price breaks above 20-period Donchian high AND price > 1w EMA50 AND 1d volume > 1.5x 20-period average
# Short when: Price breaks below 20-period Donchian low AND price < 1w EMA50 AND 1d volume > 1.5x 20-period average
# Exit when price touches opposite Donchian level (20-period low for long, high for short)
# Donchian channels provide clear breakout levels
# 1w EMA50 ensures we trade with the higher timeframe trend
# Volume confirmation filters out false breakouts
# Target: 30-80 total trades over 4 years (7-20/year) with discrete sizing 0.25

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    # Donchian high = max(high over last 20 periods)
    # Donchian low = min(low over last 20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_cond = volume[i] > (1.5 * vol_ma_20)
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low (opposite side)
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high (opposite side)
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals