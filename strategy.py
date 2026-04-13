#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d ATR regime filter and volume spike confirmation
    # Long: price > Donchian(20) high AND ATR(14)/ATR(50) > 1.2 (high vol regime) AND volume > 1.5x avg
    # Short: price < Donchian(20) low AND ATR(14)/ATR(50) > 1.2 AND volume > 1.5x avg
    # Exit: opposite Donchian breakout or ATR ratio < 0.8 (low vol exit)
    # Using 4h timeframe for optimal trade frequency, Donchian for structure,
    # ATR ratio for volatility regime (avoid low vol whipsaws), volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with min_periods
    def calculate_atr(data, period):
        atr = np.full_like(data, np.nan)
        if len(data) < period:
            return atr
        # First value: simple average
        atr[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            atr[i] = (atr[i-1] * (period-1) + data[i]) / period
        return atr
    
    atr_14 = calculate_atr(tr, 14)
    atr_50 = calculate_atr(tr, 50)
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    
    # Align daily ATR ratio to 4h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian breakout signals
    donchian_long = close > highest_high
    donchian_short = close < lowest_low
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_long[i]) or 
            np.isnan(donchian_short[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: high volatility (ATR ratio > 1.2)
        high_vol_regime = atr_ratio_aligned[i] > 1.2
        low_vol_exit = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + high vol regime + volume confirmation
        long_entry = donchian_long[i] and high_vol_regime and vol_confirm
        short_entry = donchian_short[i] and high_vol_regime and vol_confirm
        
        # Exit logic: opposite Donchian breakout OR low vol regime OR volume dry-up
        long_exit = donchian_short[i] or low_vol_exit or not vol_confirm
        short_exit = donchian_long[i] or low_vol_exit or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0