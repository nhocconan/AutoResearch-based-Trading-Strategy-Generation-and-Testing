#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 12h volume confirmation and ADX trend filter.
# Long when price breaks above R1 with volume > 1.5x 12h average AND ADX > 25 (trending market).
# Short when price breaks below S1 with volume > 1.5x 12h average AND ADX > 25.
# Uses discrete position size 0.25. ATR-based stoploss (2*ATR) and opposite pivot level exit.
# Camarilla levels provide structured support/resistance; volume confirms institutional interest;
# ADX ensures we trade in trending conditions to avoid chop. Target: 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (from previous day) ===
    # Calculate daily pivot from 1d data, then align to 4h
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels: based on previous day's high, low, close
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_h_1d = np.roll(h_1d, 1)
    prev_l_1d = np.roll(l_1d, 1)
    prev_c_1d = np.roll(c_1d, 1)
    prev_h_1d[0] = np.nan
    prev_l_1d[0] = np.nan
    prev_c_1d[0] = np.nan
    
    # Camarilla calculations
    range_1d = prev_h_1d - prev_l_1d
    pivot_1d = (prev_h_1d + prev_l_1d + prev_c_1d) / 3
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (wait for daily close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 12h Indicators: Volume Confirmation ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 12h Indicators: ADX for Trend Filter ===
    # Calculate ADX from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = pd.Series(low_12h).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_12h = 100 * (plus_dm_14 / tr_14)
    minus_di_12h = 100 * (minus_dm_14 / tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 4h ATR for stoploss ===
    tr1_4h = pd.Series(high).diff()
    tr2_4h = pd.Series(low).diff().abs()
    tr3_4h = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr_4h[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_12h_aligned[i]
        atr_val = atr_4h[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 (opposite pivot level)
            if price < s1:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (opposite pivot level)
            if price > r1:
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
            # LONG: Price breaks above R1 AND volume spike AND ADX > 25 (trending)
            if (price > r1 and vol_spike and adx_val > 25):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below S1 AND volume spike AND ADX > 25 (trending)
            elif (price < s1 and vol_spike and adx_val > 25):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ADXFilter_V1"
timeframe = "4h"
leverage = 1.0