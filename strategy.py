#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# - 1d EMA(50) defines trend direction (long when price > EMA50, short when price < EMA50)
# - 4h Donchian(20) breakout for entry: long on upper band break, short on lower band break
# - 1d volume > 1.5x 20-period average for conviction
# - Exit on opposite Donchian band touch or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 4h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled to 4h)
        # 1d volume represents ~6 4h bars, so scale down
        vol_threshold = vol_ma_1d_aligned[i] / 6.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA50) + Donchian upper break + volume
            if close[i] > ema_50_1d_aligned[i] and high[i] > donchian_upper[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA50) + Donchian lower break + volume
            elif close[i] < ema_50_1d_aligned[i] and low[i] < donchian_lower[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian lower touch or trend reversal
            if low[i] <= donchian_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian upper touch or trend reversal
            if high[i] >= donchian_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals