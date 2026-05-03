#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar average and close > 1d EMA50 (uptrend)
# Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar average and close < 1d EMA50 (downtrend)
# Exit when price crosses the Donchian(20) midpoint or trend fails (close crosses 1d EMA50)
# Donchian channels provide clear structure, EMA50 filters trend direction, volume confirms breakout strength.
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.30) to minimize fee churn.
# Works in bull (buy breakouts) and bear (sell breakdowns) by following the 1d EMA50 trend.

name = "12h_Donchian20_Volume_1dEMA50_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels using 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, lookback, 20) + 1  # EMA50(1d) + Donchian(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian(20) high with volume spike and close > 1d EMA50 (uptrend)
            if (close[i] > highest_high[i] and 
                volume_spike[i] and close[i] > ema_50_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian(20) low with volume spike and close < 1d EMA50 (downtrend)
            elif (close[i] < lowest_low[i] and 
                  volume_spike[i] and close[i] < ema_50_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian(20) midpoint or close < 1d EMA50 (trend failure)
            if (close[i] < donchian_mid[i] or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(20) midpoint or close > 1d EMA50 (trend failure)
            if (close[i] > donchian_mid[i] or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals