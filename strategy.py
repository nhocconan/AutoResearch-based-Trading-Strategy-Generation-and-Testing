#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation + ATR stoploss.
# Long when price breaks above Donchian(20) high AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Uses discrete position size 0.30. Donchian provides structure, ADX filters for trending markets only (avoids chop),
# volume confirmation ensures breakout validity. Designed to capture strong trends in both bull and bear markets.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and fee drag.

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
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX (14-period) for trend filter ===
    # Calculate True Range
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3_1d = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Directional Movement
    dm_plus = pd.Series(np.where((high_1d - pd.Series(high_1d).shift(1)) > (pd.Series(low_1d).shift(1) - low_1d),
                                 np.maximum(high_1d - pd.Series(high_1d).shift(1), 0), 0))
    dm_minus = pd.Series(np.where((pd.Series(low_1d).shift(1) - low_1d) > (high_1d - pd.Series(high_1d).shift(1)),
                                  np.maximum(pd.Series(low_1d).shift(1) - low_1d, 0), 0))
    
    # Smooth DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_smooth = atr_1d  # already smoothed
    
    # Calculate Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d_smooth
    di_minus = 100 * dm_minus_smooth / atr_1d_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian, 30 for ADX)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        adx_val = adx_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Stoploss: price < entry_price - 2.0 * ATR
            if price < entry_price - 2.0 * atr_val:
                exit_signal = True
            # Exit if ADX falls below 20 (trend weakening) or volume spike ends
            elif adx_val < 20 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Stoploss: price > entry_price + 2.0 * ATR
            if price > entry_price + 2.0 * atr_val:
                exit_signal = True
            # Exit if ADX falls below 20 (trend weakening) or volume spike ends
            elif adx_val < 20 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper Donchian channel AND ADX > 25 (trending) AND volume spike
            if price > upper_channel and adx_val > 25 and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price breaks below lower Donchian channel AND ADX > 25 (trending) AND volume spike
            elif price < lower_channel and adx_val > 25 and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_1dADX_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0