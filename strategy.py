#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike filter and ATR regime filter.
# Long when price breaks above Camarilla R1 (1-day) AND 1d volume > 2.0x 20-period average AND ATR(14) > ATR(50) (expanding volatility).
# Short when price breaks below Camarilla S1 (1-day) AND 1d volume > 2.0x 20-period average AND ATR(14) > ATR(50).
# Exit when price returns to Camarilla pivot point (PP) or ATR contracts (ATR(14) < ATR(50)).
# Uses discrete position size 0.25. Camarilla provides intraday support/resistance, 1d volume confirms institutional interest,
# ATR regime filter ensures breakouts occur in expanding volatility environments.
# Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla levels (R1, S1, PP) and volume MA ===
    # Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # PP = (high + low + close)/3
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = close_1d + 1.1 * rng / 12.0
    camarilla_s1 = close_1d - 1.1 * rng / 12.0
    
    # Align Camarilla levels to 4h timeframe (1-day lag for completed bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d volume moving average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Get 4h data for ATR calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range for ATR calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime filter
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_4h, atr_50)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_14_val = atr_14_aligned[i]
        atr_50_val = atr_50_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or ATR contracts
            if price <= pp_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or ATR contracts
            if price >= pp_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: 1d volume > 2.0x 20-period average
            vol_filter = vol > 2.0 * vol_ma_val
            
            # ATR regime filter: expanding volatility (ATR(14) > ATR(50))
            atr_filter = atr_14_val > atr_50_val
            
            # LONG: price breaks above Camarilla R1 with volume and ATR confirmation
            if price > r1_val and vol_filter and atr_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S1 with volume and ATR confirmation
            elif price < s1_val and vol_filter and atr_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR1S1_1dVolumeSpike_ATRRegime_V1"
timeframe = "4h"
leverage = 1.0