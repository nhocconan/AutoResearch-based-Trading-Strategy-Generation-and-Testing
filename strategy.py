#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and ADX(14) trend filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending up).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending down).
# Uses discrete position size 0.25. Camarilla levels from 1d provide institutional reference points.
# Volume spike confirms institutional participation. ADX filter ensures we only trade in trending regimes
# to avoid whipsaws in ranging markets. Designed to capture medium-term trends in both bull and bear markets.
# Target: 80-120 trades over 4 years (20-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Camarilla levels based on previous day's OHLC
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_r1 = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1 = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align HTF indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for ADX, 20 for volume MA)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below Camarilla S1 or ADX drops below 20 (trend weakening)
            if price < s1_level or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above Camarilla R1 or ADX drops below 20 (trend weakening)
            if price > r1_level or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND ADX > 25 (strong uptrend)
            if price > r1_level and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND ADX > 25 (strong downtrend)
            elif price < s1_level and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dVolumeSpike_ADXFilter_V1"
timeframe = "12h"
leverage = 1.0