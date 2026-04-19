#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour volume confirmation and daily ATR filter.
# Long when: price breaks above Donchian upper channel AND 12h volume > 1.5x 20-period average AND 1d ATR < 0.05 * price (low volatility regime).
# Short when: price breaks below Donchian lower channel AND 12h volume > 1.5x 20-period average AND 1d ATR < 0.05 * price.
# Exit when: price crosses back through the Donchian midline (average of upper/lower).
# Uses Donchian for trend-following structure, 12h volume for institutional confirmation, 1d ATR filter to avoid chop.
# Target: 20-40 trades/year per symbol with controlled risk.
name = "4h_Donchian_Volume_ATRFilter"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 12-hour volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h_20)
    
    # 1-day ATR for volatility filter (low vol regime)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.inf  # First bar has no previous close
    tr2[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_12h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_ma = vol_12h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period average
        vol_spike = volume[i // 48] > 1.5 * vol_12h_ma if i // 48 < len(df_12h) else False  # Approximate 12h volume from 4h data
        
        # ATR filter: low volatility regime (ATR < 5% of price)
        low_vol = atr_1d_val < 0.05 * price if price > 0 else False
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume spike + low vol
            if price > high_20[i] and vol_spike and low_vol:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + volume spike + low vol
            elif price < low_20[i] and vol_spike and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals