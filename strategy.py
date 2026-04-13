#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
    # Long: price > Donchian high(20) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5x 20-period avg
    # Short: price < Donchian low(20) AND 1d ATR(14) > 1d ATR(50) AND volume > 1.5x 20-period avg
    # Exit: opposite Donchian breakout or volatility contraction
    # Using 12h primary timeframe for low trade frequency, Donchian for structure,
    # 1d ATR regime filter to avoid low-volatility whipsaws, volume for confirmation.
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
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with Wilder's smoothing
    def atr_wilder(tr, period):
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        # First value is simple average
        atr[period-1] = np.mean(tr[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = atr_wilder(tr, 14)
    atr_50 = atr_wilder(tr, 50)
    
    # Regime filter: ATR(14) > ATR(50) = expanding volatility (good for breakouts)
    vol_expanding = atr_14 > atr_50
    
    # Align daily ATR regime to 12h
    vol_expanding_aligned = align_htf_to_ltf(prices, df_1d, vol_expanding.astype(float))
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian breakout signals
    donchian_breakout_up = close > highest_high  # Price above upper channel
    donchian_breakout_dn = close < lowest_low    # Price below lower channel
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_expanding_aligned[i]) or np.isnan(donchian_breakout_up[i]) or 
            np.isnan(donchian_breakout_dn[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: expanding volatility
        vol_regime = vol_expanding_aligned[i] > 0.5  # Boolean as float
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + vol regime + volume confirmation
        long_entry = donchian_breakout_up[i] and vol_regime and vol_confirm
        short_entry = donchian_breakout_dn[i] and vol_regime and vol_confirm
        
        # Exit logic: opposite Donchian breakout or volatility contraction
        long_exit = donchian_breakout_dn[i] or not vol_regime
        short_exit = donchian_breakout_up[i] or not vol_regime
        
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

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0