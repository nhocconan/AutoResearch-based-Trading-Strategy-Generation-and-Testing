#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for structural trend bias (long when price > EMA50, short when price < EMA50)
# Donchian(20) breakout provides entry timing in direction of 12h trend
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~19-50 trades/year per symbol with 0.28 sizing
# 12h EMA50 filter reduces false breakouts in choppy markets while capturing strong trends
# Works in both bull and bear markets by following the dominant 12h trend

name = "4h_Donchian20_12hEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 51:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels from previous 4h bar
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    
    # Donchian upper = highest high of previous 20 bars
    # Donchian lower = lowest low of previous 20 bars
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        window_high = high_prev[i-20+1:i+1]
        window_low = low_prev[i-20+1:i+1]
        if not np.any(np.isnan(window_high)) and not np.any(np.isnan(window_low)):
            donchian_high[i] = np.max(window_high)
            donchian_low[i] = np.min(window_low)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 12h data for EMA50 (51 periods) + Donchian needs 20 bars + volume EMA20
    start_idx = max(51, 20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50: long above EMA50, short below EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Donchian breakout above with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.28
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Donchian breakdown below with volume spike
                if close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.28
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below (failure of breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above (failure of breakdown)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals