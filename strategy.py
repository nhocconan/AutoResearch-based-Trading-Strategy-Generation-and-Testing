#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volatility expansion and volume confirmation
# Targets breakouts during volatility expansion phases in both bull and bear markets
# Uses 1d ATR and volume filters to ensure breakouts occur with institutional participation
# Position size: 0.25 to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volatility and volume filters ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 20-period average ATR on daily data
    atr_ma20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma20_1d)
    
    # 1d volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # === 4h Donchian channel (20-period) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma20_1d_aligned[i]) or \
           np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1d ATR > 20-period average ATR
        vol_filter = atr_1d_aligned[i] > atr_ma20_1d_aligned[i]
        
        # Volume filter: current 1d volume > 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > volume_ma20_1d_aligned[i]
        
        # Combined filter: need both volatility expansion and volume confirmation
        filter_ok = vol_filter and volume_filter
        
        if position == 0:
            # Long entry: price breaks above Donchian high with filter confirmation
            if close[i] > donchian_high[i-1] and filter_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with filter confirmation
            elif close[i] < donchian_low[i-1] and filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or filter fails
            if close[i] < donchian_low[i-1] or not filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or filter fails
            if close[i] > donchian_high[i-1] or not filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0