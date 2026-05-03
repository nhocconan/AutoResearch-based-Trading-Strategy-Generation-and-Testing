#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long: Close breaks above Donchian upper AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-period MA
# Short: Close breaks below Donchian lower AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Donchian breakout or price crosses 1d EMA50 or volume drops.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure; 1d EMA50 filters for trend direction only; volume confirmation
# reduces false breakouts. Works in bull via long signals and bear via short signals when aligned with trend.

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Donchian upper AND price > 1d EMA50 (uptrend) AND volume spike
            if close_val > donchian_upper[i] and close_val > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Donchian lower AND price < 1d EMA50 (downtrend) AND volume spike
            elif close_val < donchian_lower[i] and close_val < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Donchian lower OR price < 1d EMA50 (trend change) OR volume drops
            if close_val < donchian_lower[i] or close_val < ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Donchian upper OR price > 1d EMA50 (trend change) OR volume drops
            if close_val > donchian_upper[i] or close_val > ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals