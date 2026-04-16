#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R1/S1 breakout with volume confirmation) and 1d ADX filter.
# Long when price breaks above 12h Camarilla R1 AND volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below 12h Camarilla S1 AND volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit on opposite Camarilla level break (R1 for shorts, S1 for longs) or ATR-based stop (1.5*ATR).
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance; ADX filters for trending markets.
# Volume confirmation avoids false breakouts. Designed for 6h timeframe to capture medium-term moves with low trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 12h bar) ===
    df_12h = get_htf_data(prices, '12h')
    # Camarilla uses previous day's OHLC - for 12h, use prior 12h bar
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for current 12h bar (based on prior 12h bar)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    #          S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    range_12h = h_12h - l_12h
    camarilla_r1_12h = c_12h + (range_12h * 1.1 / 12)
    camarilla_s1_12h = c_12h - (range_12h * 1.1 / 12)
    camarilla_r1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 1d Indicators: ADX(14) for trend strength ===
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
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1d_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d_smooth
    di_minus = 100 * dm_minus_smooth / atr_1d_smooth
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h ATR for stoploss ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1_12h_aligned[i]) or np.isnan(camarilla_s1_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1_12h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1_12h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND ADX > 25 (trending)
            if price > camarilla_r1_12h_aligned[i] and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND ADX > 25 (trending)
            elif price < camarilla_s1_12h_aligned[i] and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ADXFilter_V1"
timeframe = "6h"
leverage = 1.0