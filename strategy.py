#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long: Close breaks above upper Donchian AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short: Close breaks below lower Donchian AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or EMA50 trend reversal.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide robust trend-following structure; 1w EMA50 filters higher timeframe trend;
# volume confirmation reduces false signals. Works in bull via long signals with trend alignment
# and in bear via short signals with trend alignment.

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels for 12h (using previous 20 bars' H/L)
    # Upper = max(high[-20:]), Lower = min(low[-20:])
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above upper Donchian AND uptrend AND volume spike
            if close_val > donchian_upper[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian AND downtrend AND volume spike
            elif close_val < donchian_lower[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below lower Donchian OR trend turns down
            if close_val < donchian_lower[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above upper Donchian OR trend turns up
            if close_val > donchian_upper[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals