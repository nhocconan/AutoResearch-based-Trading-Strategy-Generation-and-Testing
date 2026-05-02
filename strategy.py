#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for structure-based entries with trend alignment
# 1d EMA50 ensures trades align with higher timeframe trend to avoid whipsaws
# Volume spike (1.8x 20-period average) confirms institutional participation
# Discrete sizing 0.28 balances profit potential with fee drag (target 75-200 trades/4 years)
# Works in bull/bear by only taking breaks in direction of 1d trend

name = "4h_Donchian20_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Donchian upper breakout long: close > upper channel with 1d uptrend
            breakout_long = close[i] > high_ma[i]
            # Donchian lower breakdown short: close < lower channel with 1d downtrend
            breakout_short = close[i] < low_ma[i]
            
            # 1d EMA50 trend filter
            ema_long = close[i] > ema_50_1d_aligned[i]
            ema_short = close[i] < ema_50_1d_aligned[i]
            
            if breakout_long and ema_long and volume_spike[i]:
                signals[i] = 0.28
                position = 1
            elif breakout_short and ema_short and volume_spike[i]:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian lower channel break or trend reversal
            if close[i] < low_ma[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Donchian upper channel break or trend reversal
            if close[i] > high_ma[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals