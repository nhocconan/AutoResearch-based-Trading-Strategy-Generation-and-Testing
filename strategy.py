#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# In bull markets: buy when price breaks above Donchian upper band with ADX > 25 (uptrend).
# In bear markets: sell when price breaks below Donchian lower band with ADX > 25 (downtrend).
# Volume > 1.5x average confirms the breakout. Position size 0.25 for risk control.
# Donchian channels provide clear support/resistance levels that work in both trending and ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for ADX trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Donchian Channel (20) ===
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 1d ADX (14) for trend filter ===
    # Calculate True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1d
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower or trend weakens
            if price < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper or trend weakens
            if price > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require trending market (ADX > 25)
            if adx_val > 25:
                # Buy when price breaks above Donchian upper with volume
                if price > upper and vol_ratio > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Sell when price breaks below Donchian lower with volume
                elif price < lower and vol_ratio > 1.5:
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

name = "4h_Donchian20_ADX_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0