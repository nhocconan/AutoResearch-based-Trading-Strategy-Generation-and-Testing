#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakout with volume confirmation and ATR filter.
# Long when price breaks above 1w Donchian upper channel (20-period), volume > 1.5x 20-day average, and ATR(14) < ATR(50) (low volatility breakout).
# Short when price breaks below 1w Donchian lower channel, volume > 1.5x 20-day average, and ATR(14) < ATR(50).
# Exit when price returns to the 1w Donchian midpoint or ATR(14) > 2x ATR(50) (high volatility).
# Uses discrete position size 0.25. Donchian breakouts capture strong moves, volume confirmation avoids false breakouts,
# ATR filter ensures breakouts occur in low volatility environments (often preceding expansion). Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Donchian Channels (20-period) ===
    # Upper channel: highest high over 20 periods
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over 20 periods
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Midpoint: average of upper and lower
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_1w, midpoint_20)
    
    # Get 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume moving average (20-period) on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR (50-period) on 1d for volatility regime filter
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(midpoint_20_aligned[i]) or np.isnan(vol_ma_20_1d[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume_1d[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        midpoint = midpoint_20_aligned[i]
        vol_ma = vol_ma_20_1d[i]
        atr14 = atr_14[i]
        atr50 = atr_50[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to midpoint or volatility expands (ATR14 > 2*ATR50)
            if price <= midpoint or atr14 > 2.0 * atr50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to midpoint or volatility expands
            if price >= midpoint or atr14 > 2.0 * atr50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-day average
            vol_filter = vol > 1.5 * vol_ma
            
            # Volatility filter: ATR14 < ATR50 (low volatility breakout)
            vol_regime_filter = atr14 < atr50
            
            # LONG: price breaks above upper Donchian channel, volume spike, low vol regime
            if price > upper and vol_filter and vol_regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below lower Donchian channel, volume spike, low vol regime
            elif price < lower and vol_filter and vol_regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wDonchian20_VolumeSpike_ATRFilter_V1"
timeframe = "1d"
leverage = 1.0