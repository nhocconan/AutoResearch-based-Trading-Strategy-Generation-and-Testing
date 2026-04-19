#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# - 1d EMA(50) defines long-term trend (long when close > EMA50, short when close < EMA50)
# - 4h Donchian(20) breakout for entry: long when high breaks upper band, short when low breaks lower band
# - 1d volume > 1.5x 20-period average for conviction
# - Exit on opposite Donchian breakout or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        vol_ma_4h_scaled = vol_ma_1d_aligned[i] / 6.0
        volume_filter = vol_ma_4h_scaled > 0 and volume[i] > 1.5 * vol_ma_4h_scaled
        
        if position == 0:
            # Look for long entry: uptrend (close > 1d EMA50) + Donchian breakout + volume
            if close[i] > ema_50_1d_aligned[i] and high[i] > donchian_high[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (close < 1d EMA50) + Donchian breakout + volume
            elif close[i] < ema_50_1d_aligned[i] and low[i] < donchian_low[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian break down or trend reversal
            if low[i] < donchian_low[i-1] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian break up or trend reversal
            if high[i] > donchian_high[i-1] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals