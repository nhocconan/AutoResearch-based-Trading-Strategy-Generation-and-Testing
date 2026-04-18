#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA200 trend filter
# Donchian breakouts capture momentum with clear entry/exit levels.
# Volume confirmation filters false breakouts.
# 1d EMA200 filter ensures alignment with long-term trend to avoid counter-trend trades.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_Volume_1dEMA200"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA200 on 1d data for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema_val = ema_200_1d_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian band AND price above EMA200 AND volume spike
            if close_val > upper and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian band AND price below EMA200 AND volume spike
            elif close_val < lower and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA200 (trend change) or at lower Donchian band (mean reversion)
            if close_val < ema_val or close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA200 (trend change) or at upper Donchian band (mean reversion)
            if close_val > ema_val or close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals