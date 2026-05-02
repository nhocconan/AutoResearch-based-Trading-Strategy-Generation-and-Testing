#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band + price > 1d EMA(50) + volume > 1.5x 20-bar average
# Short when price breaks below Donchian(20) lower band + price < 1d EMA(50) + volume > 1.5x 20-bar average
# Exit on Donchian(10) opposite band touch or trend filter violation
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 12h timeframe to stay within fee drag limits

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h timeframe
    # Upper band: 20-period high, Lower band: 20-period low
    donchian_20_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit bands: 10-period for faster reversion
    donchian_10_upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian(20) and volume MA)
    start_idx = 50  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or
            np.isnan(donchian_10_upper[i]) or np.isnan(donchian_10_lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian(20) upper + price > 1d EMA + volume spike
            if close[i] > donchian_20_upper[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) lower + price < 1d EMA + volume spike
            elif close[i] < donchian_20_lower[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price touches Donchian(10) lower band OR price < 1d EMA (trend filter fails)
            if close[i] <= donchian_10_lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price touches Donchian(10) upper band OR price > 1d EMA (trend filter fails)
            if close[i] >= donchian_10_upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals