#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND choppiness < 61.8 (trending).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND choppiness < 61.8.
# Exit on opposite Donchian breakout or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Works in both bull and bear markets by requiring volume confirmation and trending regime filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Donchian(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_prev = np.roll(highest_high, 1)
    donchian_low_prev = np.roll(lowest_low, 1)
    donchian_high_prev[0] = np.nan
    donchian_low_prev[0] = np.nan
    
    # === 1d Indicators: Volume Spike and Choppiness ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Choppiness Index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(atr_1d.sum() / (highest_high_1d - lowest_low_1d)) * 14 / np.log10(14)
    chop = 100 * chop_denom
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop < 61.8  # trending regime
    
    # === 12h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h_raw = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian)
    warmup = 30
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_high_prev[i]) or
            np.isnan(donchian_low_prev[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_12h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        chop_trending = chop_filter[i]
        atr_val = atr_12h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low (opposite breakout)
            if price < donchian_low_prev[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high (opposite breakout)
            if price > donchian_high_prev[i]:
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
            # LONG: Price breaks above Donchian high AND volume spike AND trending regime
            if (price > donchian_high_prev[i] and vol_spike and chop_trending):
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Donchian low AND volume spike AND trending regime
            elif (price < donchian_low_prev[i] and vol_spike and chop_trending):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopFilter_V1"
timeframe = "12h"
leverage = 1.0