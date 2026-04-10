#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# - Donchian breakout from 20-day high/low captures momentum in both bull and bear markets
# - 1w EMA(50) ensures we trade with the weekly trend (long when price > EMA50, short when price < EMA50)
# - Volume confirmation: current 1d volume > 1.5x 20-day average to filter false breakouts
# - Designed for 1d timeframe: targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d Donchian channels (20-period)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Donchian high: highest high over past 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trend reversal) or weekly EMA flip
            if prices['close'].iloc[i] < donchian_low[i] or prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trend reversal) or weekly EMA flip
            if prices['close'].iloc[i] > donchian_high[i] or prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above Donchian high AND above weekly EMA50
                if prices['close'].iloc[i] > donchian_high[i] and prices['close'].iloc[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian low AND below weekly EMA50
                elif prices['close'].iloc[i] < donchian_low[i] and prices['close'].iloc[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals