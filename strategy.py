#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h Donchian channel breakout and 1d volume confirmation
# - 4h Donchian channel (20-period) defines trend and breakout levels
# - 1d volume > 1.3x 20-period average for conviction
# - Entry on breakout of Donchian channel in direction of 4h trend
# - Exit on opposite Donchian band touch or trend reversal
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.20 (20%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drag

name = "1h_Donchian20_4hTrend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    # 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 4h EMA(50) for trend direction (filter)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.3x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 4h EMA50) + break above Donchian high + volume
            if close[i] > ema_50_4h_aligned[i] and high[i] > donchian_high_aligned[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Look for short entry: downtrend (price < 4h EMA50) + break below Donchian low + volume
            elif close[i] < ema_50_4h_aligned[i] and low[i] < donchian_low_aligned[i] and volume_filter:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on touch of Donchian low or trend reversal
            if low[i] <= donchian_low_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on touch of Donchian high or trend reversal
            if high[i] >= donchian_high_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals