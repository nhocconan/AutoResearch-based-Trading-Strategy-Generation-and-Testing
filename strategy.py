#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume spike and chop regime filter.
# Long when price breaks above Camarilla R1 AND 12h volume > 1.5x 20-period average AND chop < 61.8 (trending).
# Short when price breaks below Camarilla S1 AND 12h volume > 1.5x 20-period average AND chop < 61.8.
# Uses discrete position size 0.25. Designed to capture institutional breakouts with volume confirmation in trending markets.
# Works in both bull and bear markets by requiring chop regime filter (trending only) and volume confirmation.
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
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using previous bar's high/low/close for current bar's levels (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_upper = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_lower = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === 4h Indicators: Choppiness Index (CHOP) < 61.8 (trending regime) ===
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # We use simplified version: CHOP < 61.8 = trending
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum()
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    chop_values = chop.values
    chop_filter = chop_values < 61.8  # trending regime
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss (optional - using Camarilla break for exit)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_filter[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = chop_filter[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_lower[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_upper[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND trending regime
            if price > camarilla_upper[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND trending regime
            elif price < camarilla_lower[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Camarilla_R1S1_12hVolumeSpike_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0