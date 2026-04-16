#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1w volume spike and 1d ADX trend filter.
# Long when price breaks above 12h Camarilla R1 AND volume > 2.0x 20-period 1w average AND 1d ADX > 25.
# Short when price breaks below 12h Camarilla S1 AND volume > 2.0x 20-period 1w average AND 1d ADX > 25.
# Exit when price crosses the 12h Camarilla midpoint (H4/L4) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture intraday breakouts in strong trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 12h bar) ===
    # Calculate from previous completed 12h bar to avoid look-ahead
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    close_12h = get_htf_data(prices, '12h')['close'].values
    
    # Camarilla levels: based on prior day's range
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2 (midpoint for exit)
    range_12h = high_12h - low_12h
    camarilla_r1_12h = close_12h + 1.1 * range_12h / 12
    camarilla_s1_12h = close_12h - 1.1 * range_12h / 12
    camarilla_h4_12h = close_12h + 1.1 * range_12h / 2  # upper midpoint
    camarilla_l4_12h = close_12h - 1.1 * range_12h / 2  # lower midpoint
    camarilla_mid_12h = (camarilla_h4_12h + camarilla_l4_12h) / 2  # same as close_12h
    
    # Align 12h Camarilla levels to 12h timeframe (no additional delay needed as based on prior bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_s1_12h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), camarilla_mid_12h)
    
    # === 1w Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    # === 1d Indicators: ADX > 25 (strong trending market filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 12h ATR for stoploss (using 12h data)
    high_12h_for_atr = get_htf_data(prices, '12h')['high'].values
    low_12h_for_atr = get_htf_data(prices, '12h')['low'].values
    close_12h_for_atr = get_htf_data(prices, '12h')['close'].values
    
    tr1_12h = pd.Series(high_12h_for_atr).diff()
    tr2_12h = pd.Series(low_12h_for_atr).diff().abs()
    tr3_12h = pd.Series(close_12h_for_atr).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), atr_12h_raw)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or np.isnan(atr_12h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_12h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla midpoint (H4/L4)
            if price < camarilla_mid_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla midpoint (H4/L4)
            if price > camarilla_mid_aligned[i]:
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
            # LONG: Price breaks above Camarilla R1 AND volume spike AND strong trending market
            if price > camarilla_r1_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND strong trending market
            elif price < camarilla_s1_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R1S1_1wVolumeSpike_1dADX_V1"
timeframe = "12h"
leverage = 1.0