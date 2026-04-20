#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter
# Long when price breaks above 20-period Donchian high with 12h volume spike and ADX > 25 (trending)
# Short when price breaks below 20-period Donchian low with 12h volume spike and ADX > 25 (trending)
# Avoids false breakouts in ranging markets via ADX filter. Volume confirms institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume spike (> 2.0 x 20-period average)
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_spike = vol_12h > (vol_ma_20 * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_spike)
    
    # Calculate 1d ADX (14-period) for trend strength
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h ATR for stop sizing
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = vol_spike_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, trending (ADX > 25)
            if price > upper and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low, volume spike, trending (ADX > 25)
            elif price < lower and vol_spike and adx_val > 25:
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

name = "4h_Donchian20_12hVolSpike_1dADXFilter"
timeframe = "4h"
leverage = 1.0