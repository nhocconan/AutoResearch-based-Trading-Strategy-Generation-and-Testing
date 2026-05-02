#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses 6h primary timeframe to reduce trade frequency vs lower TFs while maintaining sufficient signals
# Donchian channels provide clear breakout levels based on recent price extremes
# 12h EMA50 ensures trades align with intermediate-term trend, reducing whipsaws in ranging markets
# Volume spike (2.0x 20-period average) confirms breakout validity and institutional participation
# Designed for BTC/ETH with tight entries to minimize fee drag while capturing sustained moves
# Target: 50-150 total trades over 4 years (12-37/year) within proven 6h winning range

name = "6h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian(20) channels from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: 20-period high
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel + price > 12h EMA50 + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel + price < 12h EMA50 + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower channel (reversal signal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper channel (reversal signal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals