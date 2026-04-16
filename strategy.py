#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d ADX regime filter.
# Long when price breaks above 6h Camarilla R4 level AND 12h volume > 1.5x 20-period average AND 1d ADX > 20.
# Short when price breaks below 6h Camarilla S4 level AND 12h volume > 1.5x 20-period average AND 1d ADX > 20.
# Exit when price returns to 6h Camarilla midpoint (R3/S3 average) or when 12h volume < 0.8x 20-period average (volatility contraction).
# Uses discrete position size 0.25. Camarilla provides intraday support/resistance structure, volume confirmation reduces false breakouts,
# and 1d ADX ensures we only trade when there's sufficient intraday trend strength. Target: 80-180 total trades over 4 years (20-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # === 6h Indicators: Camarilla pivot levels (based on prior 6h bar) ===
    # Camarilla levels calculated from prior period's high, low, close
    # R4 = Close + (High - Low) * 1.5
    # R3 = Close + (High - Low) * 1.25
    # S3 = Close - (High - Low) * 1.25
    # S4 = Close - (High - Low) * 1.5
    # Midpoint = (R3 + S3) / 2
    camarilla_high = pd.Series(high_6h).shift(1).values  # prior period high
    camarilla_low = pd.Series(low_6h).shift(1).values   # prior period low
    camarilla_close = pd.Series(close_6h).shift(1).values # prior period close
    
    camarilla_high[0] = camarilla_high[1] if len(camarilla_high) > 1 else camarilla_high[0]
    camarilla_low[0] = camarilla_low[1] if len(camarilla_low) > 1 else camarilla_low[0]
    camarilla_close[0] = camarilla_close[1] if len(camarilla_close) > 1 else camarilla_close[0]
    
    range_6h = camarilla_high - camarilla_low
    r4 = camarilla_close + range_6h * 1.5
    r3 = camarilla_close + range_6h * 1.25
    s3 = camarilla_close - range_6h * 1.25
    s4 = camarilla_close - range_6h * 1.5
    midpoint = (r3 + s3) / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    midpoint_aligned = align_htf_to_ltf(prices, df_6h, midpoint)
    
    # Get 12h data once before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Get 1d data once before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for regime filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(midpoint_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        midpoint_val = midpoint_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average (using 12h volume MA)
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # Regime filter: 1d ADX > 20 (trending regime)
        regime_filter = adx_val > 20
        
        # Volatility exit filter: volume < 0.8x 20-period average (contraction)
        vol_exit_filter = vol < 0.8 * vol_ma_val if vol_ma_val > 0 else True
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Camarilla midpoint OR volatility contraction
            if price <= midpoint_val or vol_exit_filter:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Camarilla midpoint OR volatility contraction
            if price >= midpoint_val or vol_exit_filter:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Camarilla R4 with volume and regime confirmation
            if price > r4_val and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S4 with volume and regime confirmation
            elif price < s4_val and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_CamarillaR4_S4_12hVolumeSpike_1dADXRegime_V1"
timeframe = "6h"
leverage = 1.0