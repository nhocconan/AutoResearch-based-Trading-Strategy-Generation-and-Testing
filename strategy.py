#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 12h volume > 1.5x 20-period average AND 12h chop > 61.8 (ranging market).
# Short when price breaks below Donchian(20) low AND 12h volume > 1.5x 20-period average AND 12h chop > 61.8.
# Exit when price reverts to Donchian(20) midpoint or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Works in both bull and bear markets by requiring ranging conditions (chop>61.8) and volume confirmation, avoiding strong trends.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # === 4h Indicators: ATR (14-period) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 12h Indicators: Choppiness Index (14-period) > 61.8 (ranging market) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1_12h = pd.Series(high_12h).diff()
    tr2_12h = pd.Series(low_12h).diff().abs()
    tr3_12h = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR over n periods) / (max(high-n) - min(low-n))) / log10(n)
    sum_atr_12h = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_h_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_l_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denom = max_h_12h - min_l_12h
    chop_12h = 100 * np.log10(sum_atr_12h / chop_denom) / np.log10(14)
    chop_12h = np.where(chop_denom == 0, 50.0, chop_12h)  # avoid division by zero
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    chop_filter = chop_12h_aligned > 61.8  # ranging market
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or
            np.isnan(atr_4h_raw[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_ranging = chop_filter[i]
        atr_val = atr_4h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to Donchian midpoint
            if price <= donch_mid[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to Donchian midpoint
            if price >= donch_mid[i]:
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
            # LONG: Price breaks above Donchian high AND volume spike AND ranging market
            if price > donch_high[i] and vol_spike and is_ranging:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike AND ranging market
            elif price < donch_low[i] and vol_spike and is_ranging:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_12hVolumeSpike_12hChop_V1"
timeframe = "4h"
leverage = 1.0