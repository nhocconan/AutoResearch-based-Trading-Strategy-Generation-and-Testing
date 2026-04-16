#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 (uptrend) AND volume > 1.8x 20-period average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 (downtrend) AND volume > 1.8x 20-period average.
# Uses ATR-based trailing stop (exit when price moves against position by 2.5x ATR).
# Designed to capture strong trends in both bull and bear markets with tight entries to minimize fee drag.
# Target: 100-180 trades over 4 years (25-45/year) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR (14-period) for trailing stop ===
    tr1 = pd.Series(high).rolling(window=2, min_periods=2).max().values - pd.Series(low).rolling(window=2, min_periods=2).min().values
    tr2 = np.abs(pd.Series(high).rolling(window=2, min_periods=2).max().values - pd.Series(close).shift(1).rolling(window=2, min_periods=2).min().values)
    tr3 = np.abs(pd.Series(low).rolling(window=2, min_periods=2).min().values - pd.Series(close).shift(1).rolling(window=2, min_periods=2).max().values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume MA, 14 for ATR)
    warmup = 60
    
    # Track position state and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long positions
    lowest_since_entry = 0.0   # for short positions
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        atr_val = atr[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Update trailing stop levels
        if position == 1:  # Long position
            highest_since_entry = max(highest_since_entry, price)
            # Exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, price)
            # Exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian(20) high AND price > 1d EMA50 (uptrend) AND volume spike
            if price > donchian_high and price > ema_1d and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            
            # SHORT: Price breaks below Donchian(20) low AND price < 1d EMA50 (downtrend) AND volume spike
            elif price < donchian_low and price < ema_1d and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0