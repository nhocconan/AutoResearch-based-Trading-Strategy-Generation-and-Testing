#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and ADX trend filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Camarilla level touch.
# Uses discrete position size 0.25. Works in both bull and bear markets by requiring
# volume confirmation and trend alignment via 1d ADX. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # For 4h data, we need to use daily high/low/close to calculate Camarilla levels
    # We'll use 1d data for pivot calculation and align to 4h
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    camarilla_r1_1d = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1_1d = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # === 1d Indicators: Volume Spike and ADX ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Calculate ADX on 1d data
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_4h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price touches or goes below Camarilla S1 (opposite level)
            if price <= camarilla_s1_4h[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price touches or goes above Camarilla R1 (opposite level)
            if price >= camarilla_r1_4h[i]:
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
            # LONG: Price breaks above Camarilla R1 AND volume spike AND ADX > 25 (trending)
            if (price > camarilla_r1_4h[i] and vol_spike and adx_val > 25):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND ADX > 25 (trending)
            elif (price < camarilla_s1_4h[i] and vol_spike and adx_val > 25):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_ADXFilter_V1"
timeframe = "4h"
leverage = 1.0