#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout (20) with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when price breaks below lower Donchian(20) AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Exit when price reverts to the opposite Donchian band (long exits at lower band, short exits at upper band).
# Uses discrete position size 0.25. Donchian captures volatility-based breakouts, effective in both trending and ranging markets with volume confirmation.
# 1d EMA50 ensures we trade with higher timeframe trend. Volume confirms breakout strength.
# Target: 80-160 trades over 4 years (20-40/year) to stay within fee drag limits.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_window = 20
    upper_dc = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 12h Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_dc[i]
        lower_val = lower_dc[i]
        vol_ma_val = vol_ma_20[i]
        ema50_val = ema50_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period average
        vol_filter = vol > 1.8 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below lower Donchian band
            if price < lower_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above upper Donchian band
            if price > upper_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian AND price > 1d EMA50 AND volume confirmation
            if price > upper_val and price > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower Donchian AND price < 1d EMA50 AND volume confirmation
            elif price < lower_val and price < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0