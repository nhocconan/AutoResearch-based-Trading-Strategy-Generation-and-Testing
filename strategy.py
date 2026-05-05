#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above 1h Donchian upper (20) AND price > 4h EMA50 (uptrend) AND volume spike
# Short when price breaks below 1h Donchian lower (20) AND price < 4h EMA50 (downtrend) AND volume spike
# 4h EMA50 provides smoother trend filter than shorter EMAs, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 1h (as required)

name = "1h_Donchian20_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Donchian channels (20-period)
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = high_roll_max
        donchian_lower = low_roll_min
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # Volume confirmation on 1h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND strong uptrend (price > 4h EMA50) AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND strong downtrend (price < 4h EMA50) AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian lower OR closes below 4h EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Donchian upper OR closes above 4h EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals