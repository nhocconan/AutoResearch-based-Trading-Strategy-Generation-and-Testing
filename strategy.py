#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Aroon Oscillator + 12h ADX trend filter + volume confirmation
# Aroon Oscillator (AO) measures trend strength: AO > 0 = uptrend, AO < 0 = downtrend.
# ADX > 25 confirms strong trend, filtering out ranging markets.
# Volume spike (>1.5x 20-period average) ensures momentum behind moves.
# Enter long when AO crosses above 0 with ADX > 25 and volume confirmation.
# Enter short when AO crosses below 0 with ADX > 25 and volume confirmation.
# Exit when AO crosses back through 0 or ADX falls below 20.
# Targets 10-25 trades per year (~40-100 total over 4 years) to minimize fee drag.

name = "6h_AroonOscillator_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Aroon Oscillator: Aroon Up - Aroon Down
    # Aroon Up = ((period - periods since highest high) / period) * 100
    # Aroon Down = ((period - periods since lowest low) / period) * 100
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Periods since highest high
        highest_high_idx = np.argmax(high[i-period:i+1])
        periods_since_high = period - highest_high_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low[i-period:i+1])
        periods_since_low = period - lowest_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 12h indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 30)  # Need enough data for Aroon and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(aroon_osc[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ao_val = aroon_osc[i]
        adx_val = adx_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Aroon crosses above 0, ADX > 25 (strong trend), volume confirmation
            if aroon_osc[i] > 0 and aroon_osc[i-1] <= 0 and adx_val > 25 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Aroon crosses below 0, ADX > 25 (strong trend), volume confirmation
            elif aroon_osc[i] < 0 and aroon_osc[i-1] >= 0 and adx_val > 25 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Aroon crosses below 0 or ADX falls below 20 (trend weakening)
            if aroon_osc[i] < 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Aroon crosses above 0 or ADX falls below 20 (trend weakening)
            if aroon_osc[i] > 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals