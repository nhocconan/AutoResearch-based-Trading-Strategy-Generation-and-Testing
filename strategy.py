#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1d primary timeframe with 1w HTF for trend alignment and Donchian channels for breakout signals.
# Volume confirms breakout strength. EMA34 ensures we only trade with the higher timeframe trend.
# ATR-based stoploss manages risk. Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag.
# Works in both bull and bear markets by trading breakouts only when aligned with weekly trend.

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
    
    # === 1w data (higher timeframe for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1w EMA34 for trend filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === 1d Donchian(20) channels ===
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1d volume confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (1.5 * vol_ma_20_1d)
    
    # === 1d ATR(14) for stoploss ===
    atr_1d = np.maximum(
        np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))),
        np.abs(low_1d - np.roll(close_1d, 1))
    )
    atr_ma_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_14)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20_1d[i]) or
            np.isnan(atr_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        hh = highest_high_20[i]
        ll = lowest_low_20[i]
        ema34 = ema34_1w_aligned[i]
        vol_conf = vol_confirm[i]
        atr_val = atr_ma_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low (20-period)
            if price < ll:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high (20-period)
            if price > hh:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation and trend alignment
            if vol_conf:
                # Go long when price breaks above Donchian high and price above 1w EMA34 (bullish alignment)
                if price > hh and price > ema34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low and price below 1w EMA34 (bearish alignment)
                elif price < ll and price < ema34:
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

name = "1d_Donchian20_Breakout_Volume_EMA34_1wFilter"
timeframe = "1d"
leverage = 1.0