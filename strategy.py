#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike filter and ATR trailing stop
# Uses 4h primary timeframe with 1d HTF for volume confirmation and regime.
# Volume spike on breakout adds conviction, reducing false breakouts.
# ATR trailing stop (2.0x) protects gains and limits drawdown in volatile markets.
# Camarilla R1/S1 levels provide precise intraday support/resistance from prior day.
# Works in bull markets via R1 breakouts longs and in bear markets via S1 breakdowns shorts.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume spike and Camarilla calculation) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h Camarilla levels (based on prior 1d) ===
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from prior 1d close, high, low
    cam_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    cam_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    cam_R1_aligned = align_htf_to_ltf(prices, df_1d, cam_R1)
    cam_S1_aligned = align_htf_to_ltf(prices, df_1d, cam_S1)
    
    # === 1d Volume Spike Filter ===
    # Volume > 1.5x 20-period EMA = significant participation
    vol_ema = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ema * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(cam_R1_aligned[i]) or 
            np.isnan(cam_S1_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike_val = vol_spike_aligned[i] > 0.5  # Boolean from aligned float
        
        # === ATR CALCULATION FOR TRAILING STOP ===
        atr_4h = np.abs(high_4h - low_4h)
        atr_ma = pd.Series(atr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
        atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from high
            if price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from low
            if price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Camarilla breakout in opposite direction) ===
        if position == 1:  # Long position
            # Exit when price breaks below Camarilla S1
            if price < cam_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Camarilla R1
            if price > cam_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike for conviction
            if vol_spike_val:
                # Long on break above R1 with volume
                if price > cam_R1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short on break below S1 with volume
                elif price < cam_S1_aligned[i]:
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

name = "4h_Camarilla_R1S1_1dVolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0