#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d volatility filter.
# Long when: price breaks above Donchian(20) high, volume > 1.5x average volume, and 1d ATR ratio < 0.8 (low volatility)
# Short when: price breaks below Donchian(20) low, volume > 1.5x average volume, and 1d ATR ratio < 0.8
# Exit when price crosses back through Donchian(20) middle or volatility increases.
# Designed for ~20-30 trades/year per symbol.
name = "4h_Donchian_Breakout_Volume_Volatility"
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
    
    # 4h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 4h volume average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 10-period SMA of ATR for ratio
    atr_ma_10 = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ratio = atr_1d / (atr_ma_10 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        donch_mid_val = donchian_mid[i]
        vol_avg_val = vol_avg[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, volume surge, low volatility
            if price > donch_high and vol > 1.5 * vol_avg_val and atr_ratio_val < 0.8:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume surge, low volatility
            elif price < donch_low and vol > 1.5 * vol_avg_val and atr_ratio_val < 0.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid or volatility increases
            if price < donch_mid_val or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid or volatility increases
            if price > donch_mid_val or atr_ratio_val > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals