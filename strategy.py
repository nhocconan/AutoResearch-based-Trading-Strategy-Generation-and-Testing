#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX regime filter.
# Long when price breaks above Donchian(20) high AND volume > 1.5x MA20 volume AND 1d ADX > 25 (trending).
# Short when price breaks below Donchian(20) low AND volume > 1.5x MA20 volume AND 1d ADX > 25.
# Exit when price crosses Donchian(20) midpoint OR ADX drops below 20 (range).
# Uses discrete position size 0.25. Donchian provides clear structure, volume confirms breakout validity,
# ADX ensures trading only in trending regimes to avoid whipsaws in chop.
# 4h timeframe targets 19-50 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian(20) and volume MA20 ===
    # Donchian high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume MA20
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = abs(pd.Series(high_1d).rolling(2).max().values - pd.Series(close_1d).shift(1).values)
    tr3 = abs(pd.Series(low_1d).rolling(2).min().values - pd.Series(close_1d).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    up_14 = pd.Series(up_move).ewm(span=14, adjust=False, min_periods=14).mean().values
    down_14 = pd.Series(down_move).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * (up_14 / tr_14)
    minus_di = 100 * (down_14 / tr_14)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian(20) + volume MA20 + ADX(14) needs ~50 bars
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        donch_mid = donchian_mid[i]
        vol_ma = volume_ma20[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midpoint OR ADX < 20 (range)
            if (price < donch_mid) or (adx_val < 20):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midpoint OR ADX < 20 (range)
            if (price > donch_mid) or (adx_val < 20):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current volume > 1.5x MA20 volume
            vol_confirm = vol > (1.5 * vol_ma)
            
            # LONG: Break above Donchian high + volume confirm + ADX > 25 (trending)
            if (price > donch_high) and vol_confirm and (adx_val > 25):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Break below Donchian low + volume confirm + ADX > 25 (trending)
            elif (price < donch_low) and vol_confirm and (adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirm_ADXRegimeFilter_V1"
timeframe = "4h"
leverage = 1.0