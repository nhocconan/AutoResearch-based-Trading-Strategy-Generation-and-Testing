#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Donchian breakout captures momentum, 1d EMA50 filters trend direction to avoid counter-trend trades.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    
    donchian_upper = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_12h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d data
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to lower timeframe (12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_upper_val = donchian_upper_aligned[i]
        donchian_lower_val = donchian_lower_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian AND price above EMA50 AND volume spike
            if close_val > donchian_upper_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian AND price below EMA50 AND volume spike
            elif close_val < donchian_lower_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA50 (trend change)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA50 (trend change)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals