#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume confirmation and 1d ADX trend filter.
# Long when price breaks above Camarilla R1 (1d) AND 1d volume > 1.5x 20-period average AND 1d ADX > 20 (trending market).
# Short when price breaks below Camarilla S1 (1d) AND 1d volume > 1.5x 20-period average AND 1d ADX > 20.
# Exit when price crosses the Camarilla midpoint (PP) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture intraday breakouts in trending markets with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.
# Works in both bull and bear markets by requiring trend (ADX>20) and volume confirmation, avoiding choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (from previous 1d bar) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's OHLC)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # PP = (high + low + close)/3
    hl_range = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * hl_range / 12
    camarilla_s1 = close_1d - 1.1 * hl_range / 12
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    
    # Align to 4h timeframe (values represent levels from previous completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1d Indicators: ADX > 20 (trending market filter) ===
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
    strong_trend = adx_aligned > 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 4h ATR for stoploss
    tr1_4h = pd.Series(high).diff()
    tr2_4h = pd.Series(low).diff().abs()
    tr3_4h = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(strong_trend[i]) or np.isnan(atr_4h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla PP
            if price < camarilla_pp_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla PP
            if price > camarilla_pp_aligned[i]:
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
            # LONG: Price breaks above Camarilla R1 AND volume spike AND trending market
            if price > camarilla_r1_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND trending market
            elif price < camarilla_s1_aligned[i] and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_1dADX_V1"
timeframe = "4h"
leverage = 1.0