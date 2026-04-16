#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and chop regime filter.
# Long when price breaks above 20-period Donchian high AND 1d volume > 1.3x 20-period average AND chop > 61.8 (ranging market).
# Short when price breaks below 20-period Donchian low AND 1d volume > 1.3x 20-period average AND chop > 61.8.
# Exit on opposite Donchian break or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 1d Indicators: Volume Spike and Choppiness Index ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume MA
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # Choppiness Index (14)
    atr_1d = []
    for i in range(len(close_1d)):
        if i < 14:
            atr_1d.append(np.nan)
        else:
            tr_1d = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
            atr_1d.append(tr_1d)
    atr_1d = np.array(atr_1d)
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_1d - lowest_low_1d
    chop_1d = 100 * np.log10(atr_sum_1d / chop_denominator) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned > 61.8  # ranging market
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_regime = chop_filter[i]
        atr_val = atr[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (mean reversion)
            if price < donchian_low[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (mean reversion)
            if price > donchian_high[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR above entry
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND volume spike AND chop regime (ranging)
            if price > donchian_high[i] and vol_spike and chop_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike AND chop regime (ranging)
            elif price < donchian_low[i] and vol_spike and chop_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0