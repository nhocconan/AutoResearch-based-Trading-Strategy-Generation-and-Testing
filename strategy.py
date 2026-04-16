#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ATR-based regime filter.
# Long when price breaks above Camarilla R3 (1d) AND 1d volume > 1.5x 20-period average AND 6h ATR(14) < 6h ATR(50) (low volatility regime).
# Short when price breaks below Camarilla S3 (1d) AND 1d volume > 1.5x 20-period average AND 6h ATR(14) < 6h ATR(50).
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance, volume confirms participation,
# ATR regime filter avoids whipsaws in high volatility. Designed to capture breakouts in both bull and bear markets.
# Target: 60-120 trades over 4 years (15-30/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: ATR for regime filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_50  # Low volatility regime
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla pivot levels (R3, S3) ===
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ATR)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        vol_spike_1d = volume_spike_1d_aligned[i]
        low_vol = low_vol_regime[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below R3 or volume spike ends or volatility increases
            if price < r3 or not vol_spike or not low_vol:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above S3 or volume spike ends or volatility increases
            if price > s3 or not vol_spike or not low_vol:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 AND 6h volume spike AND 1d volume spike AND low volatility regime
            if price > r3 and vol_spike and vol_spike_1d and low_vol:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 AND 6h volume spike AND 1d volume spike AND low volatility regime
            elif price < s3 and vol_spike and vol_spike_1d and low_vol:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dVolume_ATRRegime_V1"
timeframe = "6h"
leverage = 1.0