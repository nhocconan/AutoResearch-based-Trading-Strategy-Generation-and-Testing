#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with daily trend filter and volume confirmation
# Uses Donchian(20) channels on 12h timeframe to capture breakouts in trending markets
# Daily EMA(50) filter ensures trades align with higher timeframe trend direction
# Volume confirmation reduces false breakouts
# Designed for low frequency (target: 12-37 trades/year) to minimize fee impact
# Works in both bull/bear via trend-following logic: buy breakouts in uptrend, sell breakdowns in downtrend

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from daily EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat, look for entry
            # Long: breakout above Donchian high in uptrend with volume
            if close[i] > highest_high[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: breakdown below Donchian low in downtrend with volume
            elif close[i] < lowest_low[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
        # Exit conditions
        elif position == 1:  # Long position
            # Exit on breakdown below Donchian low or trend reversal
            if close[i] < lowest_low[i] or (close[i] < ema_50_1d_aligned[i] and not uptrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout above Donchian high or trend reversal
            if close[i] > highest_high[i] or (close[i] > ema_50_1d_aligned[i] and not downtrend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
    
    return signals