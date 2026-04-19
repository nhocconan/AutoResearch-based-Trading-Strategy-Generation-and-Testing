#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation
# - 4h Donchian(20) breakout for entry: long on upper band break, short on lower band break
# - 12h EMA(34) for trend filter: only take long when price > EMA34, short when price < EMA34
# - 4h volume > 1.5x 20-period average for confirmation
# - Exit on opposite Donchian band touch or trend reversal
# - Position size: 0.25 (25%) to balance return and drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_12hTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_4h[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x average
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Look for long entry: price breaks above upper Donchian band + uptrend + volume
            if close[i] > high_20[i] and close[i] > ema_34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below lower Donchian band + downtrend + volume
            elif close[i] < low_20[i] and close[i] < ema_34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on lower Donchian band touch or trend reversal
            if close[i] < low_20[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on upper Donchian band touch or trend reversal
            if close[i] > high_20[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals