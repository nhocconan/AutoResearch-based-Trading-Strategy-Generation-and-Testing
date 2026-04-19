#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# - 1w EMA(50) defines trend direction (long when price > EMA50, short when price < EMA50)
# - 12h Donchian channel breakout: long on upper band break, short on lower band break
# - Volume filter: 12h volume > 1.2x 20-period average for conviction
# - Exit on opposite Donchian band touch or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "12h_Donchian20_1wTrend_Volume_v1"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.2x average
        volume_filter = vol_ma[i] > 0 and volume[i] > 1.2 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: uptrend + Donchian upper break + volume
            if close[i] > ema_50_1w_aligned[i] and close[i] > donchian_upper[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + Donchian lower break + volume
            elif close[i] < ema_50_1w_aligned[i] and close[i] < donchian_lower[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian lower touch or trend reversal
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian upper touch or trend reversal
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals