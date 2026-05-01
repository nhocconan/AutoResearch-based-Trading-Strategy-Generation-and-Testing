#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 is rising AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND 12h EMA50 is falling AND volume > 1.5x 20-period average
# Exit on opposite Donchian breakout or when 12h EMA50 flips direction
# Designed for low frequency (75-200 trades over 4 years) with clear trend following logic
# Works in both bull and bear markets by using 12h EMA50 to filter only strong trends

name = "4h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # 12h HTF data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 rising/falling
    ema50_rising = np.zeros(n, dtype=bool)
    ema50_falling = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(ema50_12h_aligned[i]) and not np.isnan(ema50_12h_aligned[i-1]):
            ema50_rising[i] = ema50_12h_aligned[i] > ema50_12h_aligned[i-1]
            ema50_falling[i] = ema50_12h_aligned[i] < ema50_12h_aligned[i-1]
        else:
            ema50_rising[i] = ema50_rising[i-1]
            ema50_falling[i] = ema50_falling[i-1]
    
    # Donchian(20) channels
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND 12h EMA50 rising AND volume spike
            if close[i] > donchian_high[i] and ema50_rising[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 12h EMA50 falling AND volume spike
            elif close[i] < donchian_low[i] and ema50_falling[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR 12h EMA50 starts falling
            if close[i] < donchian_low[i] or not ema50_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR 12h EMA50 starts rising
            if close[i] > donchian_high[i] or not ema50_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals