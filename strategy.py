#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot direction filter and volume confirmation.
# Long when price breaks above Donchian upper(20) AND price > 1w pivot point AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower(20) AND price < 1w pivot point AND volume > 1.5x 20-period average.
# Exit on opposite Donchian breakout or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by requiring volume confirmation and weekly pivot alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1w Indicators: Pivot Point (using prior week OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    # Pivot point = (high + low + close) / 3
    pivot_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pivot_1w_values = pivot_1w.values
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_values)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian)
    warmup = 30
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr_6h_raw[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower (20)
            if price < donchian_lower[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper (20)
            if price > donchian_upper[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper(20) AND price > 1w pivot AND volume spike
            if (price > donchian_upper[i] and 
                price > pivot_1w_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower(20) AND price < 1w pivot AND volume spike
            elif (price < donchian_lower[i] and 
                  price < pivot_1w_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_1wPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0