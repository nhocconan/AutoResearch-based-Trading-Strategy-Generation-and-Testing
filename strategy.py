#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and chop regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.8x 20-period average AND 4h chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.8x 20-period average AND 4h chop < 61.8 (trending).
# Exit when price crosses Camarilla PP (pivot point) OR chop > 61.8 (choppy regime) OR volume drops below average.
# Uses discrete position size 0.25. Designed to capture intraday momentum within 1d structure.
# Target: 80-180 trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Chopiness Index (14) for regime filter ===
    # True Range
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_hh14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_ll14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_tr14 / (max_hh14 - min_ll14)) / np.log10(14)
    chop_values = chop.values
    
    # === 4h Indicators: Camarilla Pivot Levels (from previous bar) ===
    # Camarilla levels based on previous bar's OHLC
    # R4 = Close + ((High - Low) * 1.5/2)
    # R3 = Close + ((High - Low) * 1.25/2)
    # R2 = Close + ((High - Low) * 1.1/2)
    # R1 = Close + ((High - Low) * 1.05/2)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High - Low) * 1.05/2)
    # S2 = Close - ((High - Low) * 1.1/2)
    # S3 = Close - ((High - Low) * 1.25/2)
    # S4 = Close - ((High - Low) * 1.5/2)
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    prev_close = pd.Series(close).shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.05 / 2)
    s1 = pivot - (range_hl * 1.05 / 2)
    pp = pivot
    
    r1_values = r1.values
    s1_values = s1.values
    pp_values = pp.values
    
    # === 4h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for volume MA
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * vol_ma_1d)
    
    # Align 1d volume spike to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 34 periods needed for chop, 20 for others)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_values[i]) or np.isnan(s1_values[i]) or np.isnan(pp_values[i]) or
            np.isnan(chop_values[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        chop_val = chop_values[i]
        vol_spike_1d = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below PP OR chop > 61.8 (choppy) OR volume spike ends
            if price < pp_values[i] or chop_val > 61.8 or not vol_spike_1d:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above PP OR chop > 61.8 (choppy) OR volume spike ends
            if price > pp_values[i] or chop_val > 61.8 or not vol_spike_1d:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND 1d volume spike AND chop < 61.8 (trending)
            if price > r1_values[i] and vol_spike_1d and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND 1d volume spike AND chop < 61.8 (trending)
            elif price < s1_values[i] and vol_spike_1d and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0