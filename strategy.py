#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.3x 24-period average AND 1d chop > 61.8 (range).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.3x 24-period average AND 1d chop > 61.8 (range).
# Exit when price crosses Camarilla HLC (pivot point) OR volume drops below average OR chop < 38.2 (trend).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets across bull/bear cycles.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing high-probability reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Camarilla Pivot Levels (from previous 12h bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # HLC = (high + low + close) / 3 (pivot point)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rang = prev_high - prev_low
    R1 = prev_close + rang * 1.1 / 12
    S1 = prev_close - rang * 1.1 / 12
    HLC = (prev_high + prev_low + prev_close) / 3
    
    # === 12h Indicators: Volume Spike (volume > 1.3x 24-period average) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Get 1d data once before loop for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Choppiness Index (14) ===
    # Chop = 100 * log10(sum(ATR14) / (n * log10(highest_high - lowest_low))) / log10(n)
    # Simplified: Chop = 100 * log10(ATR14_sum / (14 * log10(HH14 - LL14))) / log10(14)
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero and log of non-positive
    range_1d = hh_1d - ll_1d
    log_range = np.log10(np.maximum(range_1d, 1e-10))
    log_atr_sum = np.log10(np.maximum(atr_1d, 1e-10))
    log_14 = np.log10(14)
    
    chop_ratio = log_atr_sum / (14 * log_range)
    chop_ratio = np.maximum(chop_ratio, 1e-10)  # Avoid division by zero or log of zero
    chop = 100 * np.log10(chop_ratio) / log_14
    chop_values = chop.values
    
    # Align 1d indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 24 periods needed for volume MA, 14 for chop)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(HLC[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # Convert back to boolean
        chop_val = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below HLC OR volume spike ends OR chop < 38.2 (trend)
            if price < HLC[i] or not vol_spike or chop_val < 38.2:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above HLC OR volume spike ends OR chop < 38.2 (trend)
            if price > HLC[i] or not vol_spike or chop_val < 38.2:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND chop > 61.8 (range)
            if price > R1[i] and vol_spike and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND chop > 61.8 (range)
            elif price < S1[i] and vol_spike and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR1S1_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0