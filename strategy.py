#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation
# - Donchian(20) breakout on 4h timeframe for breakout signals
# - 12h EMA(34) to filter trend direction (only long when price > EMA34, short when price < EMA34)
# - Volume confirmation: current 4h volume > 1.5x 20-period average volume
# - Exit on opposite Donchian breakout or trend reversal
# - Position size: 0.25 to balance risk and reward
# - Designed to capture trends while minimizing false breakouts in ranging markets
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_12hEMA34_Volume_v1"
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
    
    # Get 4h data for Donchian calculation (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    
    # Donchian(20) on 4h: upper and lower bands
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above Donchian high + uptrend + volume
            if close[i] > donchian_high[i] and close[i] > ema_34_12h_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian low + downtrend + volume
            elif close[i] < donchian_low[i] and close[i] < ema_34_12h_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown of Donchian low or trend reversal
            if close[i] < donchian_low[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout of Donchian high or trend reversal
            if close[i] > donchian_high[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals