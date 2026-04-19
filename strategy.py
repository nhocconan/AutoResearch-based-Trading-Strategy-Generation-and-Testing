#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# - 12h Donchian breakout: long when price > 20-period high, short when price < 20-period low
# - 1d volume > 1.5x 20-period average for conviction (volume spike filter)
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Exit on opposite Donchian break (10-period) or trend reversal
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "12h_Donchian20_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # 12h Donchian exit (10-period for faster exit)
    highest_high_exit = pd.Series(high).rolling(window=10, min_periods=10).max()
    lowest_low_exit = pd.Series(low).rolling(window=10, min_periods=10).min()
    donchian_high_exit = highest_high_exit.values
    donchian_low_exit = lowest_low_exit.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(donchian_high_exit[i]) or np.isnan(donchian_low_exit[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + price > 12h Donchian high + volume
            if close[i] > ema_50_1w_aligned[i] and close[i] > donchian_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + price < 12h Donchian low + volume
            elif close[i] < ema_50_1w_aligned[i] and close[i] < donchian_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on price < 12h Donchian low (10-period) or trend reversal
            if close[i] < donchian_low_exit[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on price > 12h Donchian high (10-period) or trend reversal
            if close[i] > donchian_high_exit[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals