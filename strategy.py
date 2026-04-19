#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation
# - Daily Donchian(20) breakout: long on high break, short on low break
# - Weekly EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Daily volume > 1.5x 20-day average for confirmation
# - Exit on opposite Donchian band or trend reversal
# - Designed for trending markets with proper risk control via position sizing (0.25)
# - Target: 15-25 trades/year to minimize fee drag

name = "1d_DonchianBreakout_1dVolume_1wTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Donchian(20)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1d, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Daily volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current daily volume > 1.5x 20-day average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > weekly EMA50) + Donchian high break + volume
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < weekly EMA50) + Donchian low break + volume
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian low break or trend reversal
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian high break or trend reversal
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals