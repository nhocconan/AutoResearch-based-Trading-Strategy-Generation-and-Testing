#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and price range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range components for ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with Wilder smoothing (same as RMA)
    atr_14 = np.full(len(tr), np.nan, dtype=np.float64)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Align ATR and daily range to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Get 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = np.full(len(close_4h), np.nan, dtype=np.float64)
    std_20 = np.full(len(close_4h), np.nan, dtype=np.float64)
    for i in range(19, len(close_4h)):
        sma_20[i] = np.mean(close_4h[i-19:i+1])
        std_20[i] = np.std(close_4h[i-19:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Align Bollinger Bands to 4h timeframe (already aligned, but ensuring)
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_4h, sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR (14), BB (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(daily_range_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        atr = atr_14_aligned[i]
        daily_range_val = daily_range_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        sma_20_val = sma_20_aligned[i]
        
        # Volatility filter: daily range > 1.5 * ATR (expansion phase)
        vol_filter = daily_range_val > 1.5 * atr
        
        # Bollinger Band position: distance from middle band
        bb_position = (price - sma_20_val) / (upper_bb_val - sma_20_val) if (upper_bb_val - sma_20_val) != 0 else 0
        
        if position == 0:
            # Long: price touches lower BB + volatility expansion + mean reversion setup
            if price <= lower_bb_val and vol_filter and bb_position < -0.3:
                signals[i] = size
                position = 1
            # Short: price touches upper BB + volatility expansion + mean reversion setup
            elif price >= upper_bb_val and vol_filter and bb_position > 0.3:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle BB or volatility contracts
            if price >= sma_20_val or daily_range_val < atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle BB or volatility contracts
            if price <= sma_20_val or daily_range_val < atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Bollinger_Touch_VolatilityExpansion_MeanReversion"
timeframe = "4h"
leverage = 1.0