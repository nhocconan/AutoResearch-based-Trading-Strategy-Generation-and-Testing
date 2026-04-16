#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w volume spike and chop filter
# Uses 1d primary timeframe with 1w HTF for volume regime detection.
# Camarilla R1 (resistance 1) and S1 (support 1) levels act as intraday pivot points.
# Volume spike on 1w confirms institutional participation during breakouts.
# Choppiness index filter avoids whipsaw in ranging markets (CHOP > 61.8 = range).
# ATR-based stoploss (2.0x) for risk management.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets via long breakouts above R1 and in bear markets via short breakdowns below S1.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (HTF for volume regime) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # === 1d Camarilla levels (based on previous day) ===
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1w Volume regime filter (volume spike) ===
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume_1w > (2.0 * vol_ma_20_1w)  # True when volume spikes
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime)
    
    # === 1d Choppiness index filter (avoid ranging markets) ===
    # CHOP = 100 * log10(sum(ATR, n) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # True when trending (CHOP < 61.8)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # === 1d ATR for stoploss ===
    tr = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr[0] = high_1d[0] - low_1d[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(chop_filter_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        regime_ok = vol_regime_aligned[i] and chop_filter_aligned[i]
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_val = atr_aligned[i]
            if price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_val = atr_aligned[i]
            if price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC (mean reversion to mid-point) ===
        if position == 1:  # Long position
            # Exit when price reaches Camarilla midpoint (close of previous day)
            if price <= close_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches Camarilla midpoint (close of previous day)
            if price >= close_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike and trending regime
            if regime_ok:
                # Go long when price breaks above R1 with volume
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below S1 with volume
                elif price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_1wVolumeSpike_ChopFilter"
timeframe = "1d"
leverage = 1.0