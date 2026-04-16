#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ATR filter.
# Long when price breaks above R4 with volume > 1.5x 20-period average AND ATR(14) > ATR(50) (expanding volatility).
# Short when price breaks below S4 with volume > 1.5x 20-period average AND ATR(14) > ATR(50).
# Uses discrete position size 0.25. Camarilla R4/S4 are strong breakout levels, volume confirms participation,
# ATR expansion filters for genuine breakouts vs false signals. Designed to capture strong momentum moves
# in both bull and bear markets. Target: 80-160 trades over 4 years (20-40/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: ATR for volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_expanding = atr_14 > atr_50  # volatility expanding
    
    # === 6h Indicators: Volume Spike ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla pivot levels (R4, S4) ===
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_spike = volume_spike[i]
        atr_exp = atr_expanding[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below R3 (take profit) or volatility contracts
            # R3 = close + 1.1*(high-low)
            camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
            r3 = camarilla_r3_aligned[i]
            if price < r3 or not atr_exp:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above S3 (take profit) or volatility contracts
            # S3 = close - 1.1*(high-low)
            camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
            s3 = camarilla_s3_aligned[i]
            if price > s3 or not atr_exp:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R4 with volume spike and expanding volatility
            if price > r4 and vol_spike and atr_exp:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S4 with volume spike and expanding volatility
            elif price < s4 and vol_spike and atr_exp:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_CamarillaR4S4_Breakout_1dVolume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0