#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based stoploss
# Uses 12h primary timeframe with 1d HTF for trend alignment. Donchian breakouts capture momentum bursts;
# EMA50 ensures alignment with higher timeframe trend; volume confirms breakout strength.
# Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering breakouts with 1d EMA50 trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Donchian(20) channels ===
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # === 12h volume confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_12h > (1.5 * vol_ma_20_12h)
    
    # === 12h ATR(14) for stoploss ===
    tr_12h = np.maximum(
        np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1))),
        np.abs(low_12h - np.roll(close_12h, 1))
    )
    tr_12h[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20_12h[i]) or
            np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1d_aligned[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        vol_conf = vol_confirm[i]
        atr_val = atr_12h_aligned[i]
        
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
            # Exit when price retests the lower Donchian channel
            if price <= lower_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price retests the upper Donchian channel
            if price >= upper_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation and trend alignment
            if vol_conf:
                # Go long on breakout above upper channel with bullish 1d EMA50 alignment
                if price > upper_channel and close_12h[i] > ema50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short on breakdown below lower channel with bearish 1d EMA50 alignment
                elif price < lower_channel and close_12h[i] < ema50:
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

name = "12h_Donchian20_1dEMA50_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0