#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND chop < 61.8.
# Exit on opposite Camarilla level (R1 for shorts, S1 for longs) or ATR(14) stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume confirmation in trending regimes.
# Works in both bull and bear markets by requiring volume confirmation and choppiness filter to avoid false breakouts in chop.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close + (1.1 * (high - low) / 12)
    camarilla_s1 = close - (1.1 * (high - low) / 12)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h Indicators: Choppiness Index (CHOP) < 61.8 for trending regime ===
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # Using 14-period CHOP as standard
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum()
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min
    chop = 100 * (np.log10(sum_atr_14) - np.log10(max_high_14 - min_low_14)) / np.log10(14)
    chop_values = chop.values
    chop_filter = chop_values < 61.8  # Trending regime
    
    # === 4h ATR for stoploss ===
    atr_14_raw = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for CHOP/volume MA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_filter[i]) or np.isnan(atr_14_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        atr_val = atr_14_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_s1[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_r1[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND trending regime
            if price > camarilla_r1[i] and vol_spike and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND trending regime
            elif price < camarilla_s1[i] and vol_spike and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1_S1_VolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0