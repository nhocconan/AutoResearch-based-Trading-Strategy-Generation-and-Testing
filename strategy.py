#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 4h volume confirmation and 1d chop regime filter (CHOP > 61.8 = range = mean reversion).
# Long when price breaks above Donchian upper AND volume > 1.5x 20-period average AND chop > 61.8.
# Short when price breaks below Donchian lower AND volume > 1.5x 20-period average AND chop > 61.8.
# Exit when price returns to Donchian midpoint (mean reversion in ranging markets).
# Uses discrete position size 0.25. Chop filter ensures we only mean revert in ranging markets, avoiding strong trends.
# Volume confirmation reduces false breakouts. Target: 100-180 total trades over 4 years (25-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - smoothed true range
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (log10(14) * (HH14 - LL14)))
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denom = np.log10(14) * (hh_14 - ll_14)
    # Avoid division by zero
    chop = np.where(denom > 0, 100 * np.log10(sum_tr_14 / denom), 100)
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h data for Donchian channels and volume MA
    # Donchian(20) - upper/lower/lower
    donch_hi_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lo_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_hi_20 + donch_lo_20) / 2.0
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_hi_20[i]) or np.isnan(donch_lo_20[i]) or np.isnan(donch_mid_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        donch_hi = donch_hi_20[i]
        donch_lo = donch_lo_20[i]
        donch_mid = donch_mid_20[i]
        vol_ma_val = vol_ma_20[i]
        chop_val = chop_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # Regime filter: chop > 61.8 (ranging market = mean reversion opportunity)
        regime_filter = chop_val > 61.8
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian midpoint (mean reversion)
            if price <= donch_mid:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian midpoint (mean reversion)
            if price >= donch_mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper with volume and regime confirmation
            if price > donch_hi and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower with volume and regime confirmation
            elif price < donch_lo and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_4hVolumeSpike_1dChopRegime_V1"
timeframe = "4h"
leverage = 1.0