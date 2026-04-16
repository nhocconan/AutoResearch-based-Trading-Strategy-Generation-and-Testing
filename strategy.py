#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Camarilla R3 (1d) AND 1d volume > 1.5x 20-period average AND 12h chop < 61.8 (trending).
# Short when price breaks below Camarilla S3 (1d) AND 1d volume > 1.5x 20-period average AND 12h chop < 61.8 (trending).
# Uses discrete position size 0.25. Camarilla levels provide institutional support/resistance, volume confirms breakout strength,
# chop filter ensures we only trade in trending markets. Designed for 12h timeframe to capture multi-day moves with low frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag and maximize edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Choppiness Index (14-period) ===
    # Chop = 100 * log10(sum(ATR14) / (n * (highest_high - lowest_low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (14 * (highest_high14 - lowest_low14))) / np.log10(14)
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) ===
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align 1d data to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA, 14 for chop/ATR)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(chop[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        chop_val = chop[i]
        vol_spike_12h = volume_spike[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_spike_1d = volume_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below S3 or choppiness becomes too high (range) or volume spike ends
            if price < s3 or chop_val > 61.8 or not vol_spike_12h:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above R3 or choppiness becomes too high (range) or volume spike ends
            if price > r3 or chop_val > 61.8 or not vol_spike_12h:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 AND 12h volume spike AND 1d volume spike AND trending market (chop < 61.8)
            if price > r3 and vol_spike_12h and vol_spike_1d and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 AND 12h volume spike AND 1d volume spike AND trending market (chop < 61.8)
            elif price < s3 and vol_spike_12h and vol_spike_1d and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dVolume_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0