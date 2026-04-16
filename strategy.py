#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR stoploss.
# Long when price breaks above 1d Donchian upper band with volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower band with volume > 1.5x 20-period average.
# Exit when price crosses the 1d Donchian midline (20-period average of high/low).
# Uses discrete position size 0.25. Donchian breakout provides clear structure, volume confirmation
# reduces false signals, and ATR-based stoploss manages risk. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Donchian Channel (20) ===
    # Upper band: highest high over 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midline: average of upper and lower band
    midline_20 = (upper_20 + lower_20) / 2.0
    
    # Align 1d Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midline_aligned = align_htf_to_ltf(prices, df_1d, midline_20)
    
    # Get 12h data once before loop for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Volume moving average (20-period) on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midline_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        midline_val = midline_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midline (mean reversion)
            if price < midline_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midline (mean reversion)
            if price > midline_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average (12h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # LONG: price breaks above upper Donchian band with volume confirmation
            if price > upper_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below lower Donchian band with volume confirmation
            elif price < lower_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_MidlineExit_V1"
timeframe = "12h"
leverage = 1.0