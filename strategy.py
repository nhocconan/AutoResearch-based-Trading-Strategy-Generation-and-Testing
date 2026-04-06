#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action combined with 1-day volume profile and 1-week volatility regime.
# Uses intraday mean reversion at key levels (high/low of day) filtered by weekly volatility regime.
# Designed for low-frequency trading (target: 50-150 trades over 4 years) with clear entry/exit rules.
# Works in both bull and bear markets by adapting to volatility regime.

name = "12h_priceaction1d_vol1w_volatility_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day high/low for intraday mean reversion levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's high and low levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Align daily levels to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # 1-week ATR for volatility regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr = np.maximum(
        high_1w - low_1w,
        np.maximum(
            np.abs(high_1w - np.roll(close_1w, 1)),
            np.abs(low_1w - np.roll(close_1w, 1))
        )
    )
    tr[0] = high_1w[0] - low_1w[0]  # First period
    
    # ATR(14) calculation
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1w[i] = np.mean(tr[1:14])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio (current vs 20-period average) for regime detection
    atr_ma_20 = np.full(len(atr_1w), np.nan)
    for i in range(19, len(atr_1w)):
        atr_ma_20[i] = np.mean(atr_1w[i-19:i+1])
    
    atr_ratio = atr_1w / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # 1-week volume average for institutional participation filter
    vol_1w = df_1w['volume'].values
    vol_ma_20 = np.full(len(vol_1w), np.nan)
    for i in range(19, len(vol_1w)):
        vol_ma_20[i] = np.mean(vol_1w[i-19:i+1])
    
    vol_ratio = vol_1w / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 20 periods for ATR and volume MA)
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility regime: only trade in normal to high volatility (ATR ratio > 0.8)
        volatile_enough = atr_ratio_aligned[i] > 0.8
        
        # Volume filter: institutional participation (volume > average)
        volume_filter = vol_ratio_aligned[i] > 1.0
        
        # Price position relative to previous day's levels
        price_vs_prev_high = close[i] > prev_high_aligned[i]
        price_vs_prev_low = close[i] < prev_low_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches previous day's high or stoploss
            if (price_vs_prev_high or 
                close[i] < entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches previous day's low or stoploss
            if (price_vs_prev_low or 
                close[i] > entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at daily extremes
            if volatile_enough and volume_filter:
                # Long: price near previous day's low (support)
                if close[i] <= prev_low_aligned[i] * 1.001:  # Within 0.1% of low
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price near previous day's high (resistance)
                elif close[i] >= prev_high_aligned[i] * 0.999:  # Within 0.1% of high
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals