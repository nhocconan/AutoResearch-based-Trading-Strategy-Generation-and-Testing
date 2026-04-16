#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses the opposite Donchian band (lower for longs, upper for shorts).
# Uses discrete position size 0.25. Donchian captures breakouts, 1w EMA50 ensures higher timeframe trend alignment.
# Volume confirmation filters weak breakouts. Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian(20) ===
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 1d Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data once before loop for EMA50 filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        up = upper[i]
        low_band = lower[i]
        vol_ma_val = vol_ma_20[i]
        ema50_val = ema50_1w_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band
            if price < low_band:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band
            if price > up:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND price > 1w EMA50 AND volume confirmation
            if price > up and price > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower band AND price < 1w EMA50 AND volume confirmation
            elif price < low_band and price < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter_V1"
timeframe = "1d"
leverage = 1.0