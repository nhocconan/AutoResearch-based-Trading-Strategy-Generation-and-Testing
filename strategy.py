#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# In bull regime (price > 1w EMA50), we go long on upper Donchian breakout.
# In bear regime (price < 1w EMA50), we go short on lower Donchian breakout.
# Volume spike confirms institutional participation. This structure works in both bull and bear markets
# by using the higher timeframe trend for regime filtering and Donchian breakouts for precise entry timing.
# Target: 12h timeframe with tight entries (12-37 trades/year) to minimize fee drag.

name = "12h_Donchian20_1wTrend_VolumeSpike_Regime"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 20-period Donchian channels (primary timeframe)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1w EMA50, bear if close < 1w EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: price breaks above upper Donchian channel with volume spike
            long_entry = (close_val > highest_high_20[i]) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: price breaks below lower Donchian channel with volume spike
            short_entry = (close_val < lowest_low_20[i]) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on price breaking below lower Donchian channel or regime change to bear
            if close_val < lowest_low_20[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on price breaking above upper Donchian channel or regime change to bull
            if close_val > highest_high_20[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals