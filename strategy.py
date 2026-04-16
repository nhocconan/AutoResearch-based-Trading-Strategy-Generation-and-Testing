#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR-based stop.
# Long when price breaks above 1d Donchian(20) upper band, volume > 1.5x 20-period average, and ATR(14) > 0.
# Short when price breaks below 1d Donchian(20) lower band, volume > 1.5x 20-period average, and ATR(14) > 0.
# Exit when price crosses the Donchian midline (average of upper and lower bands) or ATR drops below threshold.
# Uses discrete position size 0.25. Donchian provides clear trend structure, volume confirms breakout strength,
# ATR filter avoids low-volatility false breakouts. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian, volume MA, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    # Upper band: highest high over 20 periods
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midline: average of upper and lower bands
    midline_20 = (upper_20 + lower_20) / 2.0
    
    # Volume moving average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midline_20_aligned = align_htf_to_ltf(prices, df_1d, midline_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(midline_20_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        midline_val = midline_20_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midline or ATR too low
            if price < midline_val or atr_val < 0.001:  # avoid zero ATR
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midline or ATR too low
            if price > midline_val or atr_val < 0.001:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # ATR filter: avoid extremely low volatility
            atr_filter = atr_val > 0.001
            
            # LONG: price breaks above upper Donchian band
            if price > upper_val and vol_filter and atr_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below lower Donchian band
            elif price < lower_val and vol_filter and atr_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0