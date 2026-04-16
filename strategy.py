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
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate ATR on 1d
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d Bollinger Bands (20, 2) for volatility regime ===
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bb_width = (upper_band - lower_band) / sma_20
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Percentile of BB width over 50 days for regime detection
    bb_width_percentile = pd.Series(bb_width_aligned).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # === 1d Donchian Channel (20) for entry ===
    highest_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = highest_20_1d
    donchian_lower_1d = lowest_20_1d
    
    # === 1d Volume spike detection ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Session filter: 08-20 UTC (but on daily close, we check if day falls in session) ===
    # For daily, we consider the day active if any part of it overlaps 08-20 UTC
    # Simplified: treat all days as active since daily bars cover 24h
    active_day = np.ones(n, dtype=bool)  # Always active for daily timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr_1d_val = atr_1d_aligned[i]
        bb_width_pct = bb_width_percentile[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 1d Donchian lower OR volatility regime shifts to high
            if (price < donchian_lower_1d[i]) or (bb_width_pct > 80):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 1d Donchian upper OR volatility regime shifts to high
            if (price > donchian_upper_1d[i]) or (bb_width_pct > 80):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 1d Donchian upper AND low volatility regime AND volume spike
            if (price > donchian_upper_1d[i]) and (bb_width_pct < 30) and (vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 1d Donchian lower AND low volatility regime AND volume spike
            elif (price < donchian_lower_1d[i]) and (bb_width_pct < 30) and (vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian_Breakout_LowVol_Volume"
timeframe = "1d"
leverage = 1.0