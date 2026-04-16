#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 2.0x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 2.0x 20-period average AND chop < 61.8 (trending).
# Exit when price crosses Camarilla P (pivot point) OR chop > 61.8 (range) OR volume drops below average.
# Uses discrete position size 0.25. Designed to capture strong intraday trends with volume confirmation and regime filter.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Note: For intraday, we use previous 4h bar's high/low/close as proxy for daily
    # In practice, we'd use actual 1d data, but for 4h timeframe we approximate
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    prev_close = pd.Series(close).shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_h_l = prev_high - prev_low
    
    R1 = pivot + (range_h_l * 1.1 / 12)
    S1 = pivot - (range_h_l * 1.1 / 12)
    R1 = R1.values
    S1 = S1.values
    pivot = pivot.values
    
    # === 4h Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === 4h Indicators: Choppiness Index (CHOP) for regime filter ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Max/Min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max()
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min()
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (max_close - min_close)) / np.log10(14)
    chop = chop.values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA, 14 for CHOP)
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(pivot[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Camarilla P (pivot) OR chop > 61.8 (range) OR volume spike ends
            if price < pivot[i] or chop_val > 61.8 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Camarilla P (pivot) OR chop > 61.8 (range) OR volume spike ends
            if price > pivot[i] or chop_val > 61.8 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND chop < 61.8 (trending)
            if price > R1[i] and vol_spike and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND chop < 61.8 (trending)
            elif price < S1[i] and vol_spike and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0