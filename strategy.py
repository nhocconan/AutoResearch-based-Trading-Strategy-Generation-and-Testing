#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volatility regime filter and volume confirmation
# In bull markets: buy breakout above 20-period high when ATR ratio > 1.2 (expanding vol) with volume spike
# In bear markets: sell breakdown below 20-period low when ATR ratio > 1.2 (expanding vol) with volume spike
# ATR ratio < 0.8 = low volatility (avoid false breakouts). Volume confirms institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility regime
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_1d / atr_ma_1d  # Current ATR / 30-day average ATR
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for stop sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: 4h volume > 2.0 x 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr_ratio_aligned[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_regime = atr_ratio_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, expanding volatility (ATR ratio > 1.2), volume spike
            if price > upper and vol_regime > 1.2 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low, expanding volatility (ATR ratio > 1.2), volume spike
            elif price < lower and vol_regime > 1.2 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below Donchian low
            if price <= entry_price - 2.0 * atr_4h[i] or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above Donchian high
            if price >= entry_price + 2.0 * atr_4h[i] or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0