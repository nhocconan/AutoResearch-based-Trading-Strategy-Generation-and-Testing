#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_Breakout_Volume_Trend_1d"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2x 20-period average volume (12h)
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = vol_ma[10]  # fill beginning
    vol_ma[-10:] = vol_ma[-11]  # fill end
    vol_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high AND above daily EMA50 AND volume spike
            long_cond = (close[i] > donchian_high_aligned[i]) and (close[i] > ema50_aligned[i]) and vol_spike[i]
            
            # Short breakdown: price < Donchian low AND below daily EMA50 AND volume spike
            short_cond = (close[i] < donchian_low_aligned[i]) and (close[i] < ema50_aligned[i]) and vol_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian low OR below daily EMA50
            if (close[i] < donchian_low_aligned[i]) or (close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high OR above daily EMA50
            if (close[i] > donchian_high_aligned[i]) or (close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian breakouts with volume confirmation and trend filter (EMA50).
# Long when price breaks above 20-day high with volume spike and above daily EMA50.
# Short when price breaks below 20-day low with volume spike and below daily EMA50.
# Exits when price returns inside the channel or crosses the EMA50.
# Works in trending markets (breakouts continue) and ranging markets (mean reversion at channel extremes).
# Uses daily timeframe for structure, 12h for execution to avoid look-ahead.
# Volume spike ensures breakouts have conviction, reducing false signals.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.