#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter
# Uses 12h primary timeframe with 1d HTF for volume and choppiness regime detection.
# Camarilla pivots provide precise intraday support/resistance levels from prior 1d candle.
# Breakout above R1 or below S1 with volume confirmation indicates institutional participation.
# Choppiness regime filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion works.
# In trending markets (CHOP < 38.2), we avoid false breakouts.
# ATR-based stoploss (2.0x) and time-based exit (max 3 bars) for risk management.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via long breakouts and in bear markets via short breakdowns during ranging conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for volume and choppiness) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h Camarilla pivot levels (based on prior 12h candle) ===
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # We use the prior 12h bar's high/low/close to calculate levels for current bar
    camarilla_r1 = np.zeros_like(close_12h)
    camarilla_s1 = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        camarilla_r1[i] = close_12h[i-1] + 1.1 * (high_12h[i-1] - low_12h[i-1]) / 12
        camarilla_s1[i] = close_12h[i-1] - 1.1 * (high_12h[i-1] - low_12h[i-1]) / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 12h bar close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # === 1d Volume spike filter (expanding volume) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)  # Volume > 2x 20-period average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # === 1d Choppiness index regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,N) - min(low,N))) / log10(N)
    # CHOP > 61.8 = ranging market (good for mean reversion/breakout)
    # CHOP < 38.2 = trending market (avoid false breakouts)
    atr_1d = np.abs(high_1d - low_1d)
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid division by zero
    chop = 100 * np.log10(atr_sum_14 / chop_denominator) / np.log10(14)
    chop_regime = chop > 61.8  # True when market is ranging
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_in_trade = 0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            position = 0
            bars_in_trade = 0
            continue
        
        price = close[i]
        volume_ok = vol_spike_aligned[i]
        regime_ok = chop_regime_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position != 0:
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if position == 1 and price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
            if position == -1 and price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        # === TIME-BASED EXIT (max 3 bars) ===
        if position != 0:
            bars_in_trade += 1
            if bars_in_trade >= 3:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        # === EXIT LOGIC (mean reversion to pivot point) ===
        if position == 1:  # Long position
            # Exit when price reaches prior 12h close (pivot point)
            if price >= close_12h[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches prior 12h close (pivot point)
            if price <= close_12h[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                bars_in_trade = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and volume_ok and regime_ok:
            # Go long when price breaks above R1 with volume
            if price > camarilla_r1_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_in_trade = 1
                continue
            # Go short when price breaks below S1 with volume
            elif price < camarilla_s1_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_in_trade = 1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0