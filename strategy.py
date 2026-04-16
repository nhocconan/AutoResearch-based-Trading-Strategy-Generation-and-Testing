#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above R1 AND 1w ADX > 25 AND 1d volume > 1.3x 20-period average.
# Short when price breaks below S1 AND 1w ADX > 25 AND 1d volume > 1.3x 20-period average.
# Exit on ATR-based stoploss (2.5*ATR from entry) or opposite Camarilla break.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# strong trend (ADX>25) and volume confirmation. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla levels (based on prior 12h bar) ===
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12.0
    s1 = prev_close - camarilla_range * 1.1 / 12.0
    
    # Breakout conditions
    breakout_long = close > r1
    breakout_short = close < s1
    breakout_long_prev = np.roll(breakout_long, 1)
    breakout_short_prev = np.roll(breakout_short, 1)
    breakout_long_prev[0] = False
    breakout_short_prev[0] = False
    
    # === 1w Indicators: ADX (14-period) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1w['high']).diff()
    tr2 = pd.Series(df_1w['low']).diff().abs()
    tr3 = pd.Series(df_1w['close']).shift(1).diff().abs()
    tr_1w = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1w['high']).diff()
    dm_minus = pd.Series(df_1w['low']).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR, DM+
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # === 1d Indicators: Volume Spike ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 12h ATR for stoploss ===
    tr1_12h = pd.Series(high).diff()
    tr2_12h = pd.Series(low).diff().abs()
    tr3_12h = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
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
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_12h_raw[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 (opposite Camarilla level)
            if price < s1[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (opposite Camarilla level)
            if price > r1[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Breakout above R1 AND ADX > 25 AND volume spike
            if (breakout_long[i] and not breakout_long_prev[i] and 
                adx_val > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Breakout below S1 AND ADX > 25 AND volume spike
            elif (breakout_short[i] and not breakout_short_prev[i] and 
                  adx_val > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wADX_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0