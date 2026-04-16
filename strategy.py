#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and ATR stoploss.
# Uses 12h Donchian channel (20-period high/low) for breakout signals.
# Filters with 1d volume spike (>1.5x 20-period average) to avoid false breakouts.
# Exits on Donchian reversal or ATR-based stoploss (2x ATR).
# Position size 0.25 for risk control. Works in both bull and breakout markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 1d data (higher timeframe for volume filter) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # === 12h Donchian Channel (20) ===
    # Upper band: 20-period high
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 1d volume filter ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).shift(1) - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        atr_val = atr_12h_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian lower band
            if price < lower:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper band
            if price > upper:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above Donchian upper band with volume confirmation
            if price > upper and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: price breaks below Donchian lower band with volume confirmation
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

name = "12h_DonchianBreakout_Volume_ATRStop_V1"
timeframe = "12h"
leverage = 1.0