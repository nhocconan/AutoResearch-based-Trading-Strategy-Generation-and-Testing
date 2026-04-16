#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and ATR trailing stop.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average.
# Exit via ATR trailing stop (3*ATR from extreme) or opposite breakout.
# Uses discrete position size 0.25. Designed for fewer trades (<100/year) to minimize fee drag.
# Works in both bull and bear markets by requiring volume confirmation and using
# ATR-based trailing stop that adapts to volatility.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume Confirmation ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for trailing stop ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian)
    warmup = 30
    
    # Track position state and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_price = 0.0  # highest high for long, lowest low for short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_4h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update extreme price (highest high)
            if price > extreme_price:
                extreme_price = price
            # ATR trailing stop: 3*ATR below extreme price
            elif price < extreme_price - 3.0 * atr_val:
                exit_signal = True
            # Opposite breakout exit
            elif price < lowest_low[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update extreme price (lowest low)
            if price < extreme_price:
                extreme_price = price
            # ATR trailing stop: 3*ATR above extreme price
            elif price > extreme_price + 3.0 * atr_val:
                exit_signal = True
            # Opposite breakout exit
            elif price > highest_high[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike
            if price > highest_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike
            elif price < lowest_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_DonchianBreakout_1dVolumeSpike_ATRTrailing_V1"
timeframe = "4h"
leverage = 1.0