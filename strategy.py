#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper channel AND 12h HMA is rising AND 1d volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower channel AND 12h HMA is falling AND 1d volume > 1.5x 20-period average.
# Exit when price reverts to Donchian midline (median of upper/lower) or ATR-based stoploss (2.5*ATR from entry).
# Uses discrete position size 0.30. Designed to capture breakouts in trending markets with volume confirmation.
# Works in both bull and bear markets by requiring HMA trend alignment and volume spike, avoiding false breakouts.
# Target: 100-200 total trades over 4 years (25-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === 12h Indicators: HMA(21) for trend ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    # HMA slope: rising if current > previous, falling if current < previous
    hma_slope = np.diff(hma_12h_aligned, prepend=hma_12h_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 4h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(hma_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to Donchian midline
            if price <= donchian_mid[i]:
                exit_signal = True
            # ATR-based stoploss: 2.5*ATR below entry
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to Donchian midline
            if price >= donchian_mid[i]:
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
            # LONG: Price breaks above Donchian upper AND HMA rising AND volume spike
            if close[i] > highest_high[i] and hma_rising[i] and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian lower AND HMA falling AND volume spike
            elif close[i] < lowest_low[i] and hma_falling[i] and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_12hHMATrend_1dVolumeSpike_V1"
timeframe = "4h"
leverage = 1.0