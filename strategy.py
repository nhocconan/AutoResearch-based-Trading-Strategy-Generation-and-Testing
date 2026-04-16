#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1, S1) with volume confirmation and ATR stoploss.
# Long when price breaks above R1 with volume > 1.5x median volume (20-period).
# Short when price breaks below S1 with volume > 1.5x median volume.
# Uses discrete position size 0.25. Exits when price reaches opposite pivot level (S1 for long, R1 for short) or ATR stoploss hits (2.5x ATR).
# Camarilla pivots identify key intraday support/resistance; breakout with volume confirms institutional interest.
# 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + (1.1 * camarilla_range / 12)
    s1_1d = close_1d - (1.1 * camarilla_range / 12)
    
    # === 1d Indicators: Volume Median (20-period) ===
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # === 12h Indicators: ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (12h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    # ATR is already on primary timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(2, 20)  # Camarilla needs 2 bars, Volume median needs 20
    
    # Track position state and entry price for ATR stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_median_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_median = vol_median_aligned[i]
        atr = atr_14[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price reaches S1 (opposite pivot) OR ATR stoploss hit (2.5 * ATR below entry)
            if price <= s1 or price <= entry_price - 2.5 * atr:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price reaches R1 (opposite pivot) OR ATR stoploss hit (2.5 * ATR above entry)
            if price >= r1 or price >= entry_price + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike
            if price > r1 and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below S1 with volume spike
            elif price < s1 and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dCamarillaR1S1_Breakout_VolumeSpike1.5x_EXITopposite_ATRTrail2.5_v1"
timeframe = "12h"
leverage = 1.0