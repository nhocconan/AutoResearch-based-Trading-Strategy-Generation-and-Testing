#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 12h
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12h price range as percentage of ATR
    range_12h = high - low
    atr_safe = np.where(atr_14_1d_aligned < 1e-10, np.nan, atr_14_1d_aligned)
    range_pct = range_12h / atr_safe
    
    # Calculate 12h average true range for volatility filter
    tr_12h1 = high - low
    tr_12h2 = np.abs(high - np.roll(close, 1))
    tr_12h3 = np.abs(low - np.roll(close, 1))
    tr_12h2[0] = np.nan
    tr_12h3[0] = np.nan
    tr_12h = np.maximum(tr_12h1, np.maximum(tr_12h2, tr_12h3))
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume spike (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate 12h Donchian channel breakout (10-period)
    donchian_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 10) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(range_pct[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h range is significant relative to 1d ATR
        vol_filter = range_pct[i] > 0.8
        
        # Entry conditions: Donchian breakout with volume and volatility filter
        long_breakout = close[i] > donchian_high[i-1]
        short_breakout = close[i] < donchian_low[i-1]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and volatility
            if long_breakout and volume_spike[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and volatility
            elif short_breakout and volume_spike[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below Donchian low or volatility drops
            if close[i] < donchian_low[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above Donchian high or volatility drops
            if close[i] > donchian_high[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_1dATR_VolumeFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0