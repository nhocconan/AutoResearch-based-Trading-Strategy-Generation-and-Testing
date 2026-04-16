#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike filter and ATR trailing stop
# Uses Camarilla levels from 1d HTF for institutional reference points (R1/S1).
# Breakout above R1 or below S1 with concurrent 1d volume spike (>2.0x 20-day avg) signals institutional participation.
# ATR trailing stop (2.0x) protects gains; exit on reversion to Camarilla pivot point (PP).
# Designed for 75-150 total trades over 4 years to minimize fee drag while capturing momentum phases.
# Works in bull markets via R1 breakouts and in bear markets via S1 breakdowns during volume expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for Camarilla pivots and volume regime) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Camarilla pivot levels (R1, S1, PP) ===
    # PP = (high + low + close) / 3
    # R1 = PP + (high - low) * 1.1 / 12
    # S1 = PP - (high - low) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pp_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = pp_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d Volume regime filter (volume spike) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20_1d)  # True when volume spikes >2x average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # === 4h ATR for trailing stop ===
    atr_4h = np.maximum(high_4h - low_4h, 
                        np.absolute(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])),
                        np.absolute(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])))
    atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # === TRAILING STOPLOGIC (ATR-based) ===
        if position == 1:  # Long position
            highest_since_entry = max(highest_since_entry, price)
            if price < highest_since_entry - 2.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, price)
            if price > lowest_since_entry + 2.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (reversion to Camarilla pivot point) ===
        if position == 1:  # Long position
            # Exit when price returns to or below pivot point
            if price <= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to or above pivot point
            if price >= pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike (institutional participation)
            if vol_spike_aligned[i]:
                # Go long when price breaks above R1 with volume spike
                if price > r1_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Go short when price breaks below S1 with volume spike
                elif price < s1_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0