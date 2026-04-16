#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w ADX trend filter + volume confirmation
# In trending markets (ADX > 25), buy when price breaks above upper Donchian channel,
# sell when price breaks below lower Donchian channel. Works in bull markets (buy breakouts)
# and bear markets (sell breakdowns). Weekly ADX ensures we only trade in strong trends.
# Target: 50-100 total trades over 4 years (12-25/year) with disciplined entries.

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
    
    # === 1w data (higher timeframe for ADX trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1d Donchian Channel (20) ===
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1w ADX(14) for trend filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1w).diff()
    tr2 = abs(pd.Series(high_1w).diff())
    tr3 = abs(pd.Series(low_1w).diff())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1w).diff()
    down_move = -pd.Series(low_1w).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 1d volume ratio for confirmation ===
    vol_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = volume_1d / vol_ma_10_1d
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_1d[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_1d = np.abs(high_1d - low_1d)
            atr_ma = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_1d = np.abs(high_1d - low_1d)
            atr_ma = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below lower Donchian or trend weakens
            if price < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian or trend weakens
            if price > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25) and volume confirmation
            if adx_val > 25 and vol_ratio > 1.3:
                # Buy when price breaks above upper Donchian
                if price > upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below lower Donchian
                elif price < lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wADX_VolumeTrendFilter_v1"
timeframe = "1d"
leverage = 1.0