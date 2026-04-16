#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above R3 AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (trending).
# Short when price breaks below S3 AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (trending).
# Uses discrete position size 0.25. Camarilla levels provide institutional support/resistance, volume confirms
# institutional participation, chop filter ensures we only trade in trending markets to avoid false breakouts.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for calculations
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) ===
    # Pivot point = (high + low + close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R3 = close + 1.1 * (high - low)
    r3 = close_1d + 1.1 * (high_1d - low_1d)
    # S3 = close - 1.1 * (high - low)
    s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma)
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR = smoothed TR
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(sum_atr / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_minus_ll = hh - ll
    chop = np.where(hh_minus_ll > 0, 100 * np.log10(sum_atr / hh_minus_ll) / np.log10(14), 50)
    chop_values = chop
    
    # Align 1d indicators to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for chop, 20 for volume MA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_spike = vol_spike_aligned[i] > 0.5  # Convert back to boolean
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to pivot point or volume spike ends or chop drops
            if price <= r3_val or not vol_spike or chop_val < 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to pivot point or volume spike ends or chop drops
            if price >= s3_val or not vol_spike or chop_val < 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 AND volume spike AND chop > 61.8 (trending)
            if price > r3_val and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S3 AND volume spike AND chop > 61.8 (trending)
            elif price < s3_val and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_CamarillaR3S3_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0