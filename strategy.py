#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ADX filter and volume confirmation.
# Williams Alligator identifies trending vs ranging markets via smoothed SMAs.
# ADX from 1d confirms trend strength to avoid false signals in chop.
# Volume confirmation adds conviction to signals.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (trend up) and bear markets (trend down) by filtering with ADX.
name = "6h_WilliamsAlligator_1dADX_Volume"
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
    
    # Get daily data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMAs with future shift
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate ADX (14-period) on daily timeframe
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.concatenate([[np.nan], np.maximum(high_d[1:] - high_d[:-1], 0)])
    minus_dm = np.concatenate([[np.nan], np.maximum(low_d[:-1] - low_d[1:], 0)])
    
    # Only update DM when TR > 0
    plus_dm = np.where(tr[1:] > 0, plus_dm[1:], 0)
    minus_dm = np.where(tr[1:] > 0, minus_dm[1:], 0)
    tr = np.where(tr > 0, tr, 0)  # Avoid division by zero
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_14 = wilders_smooth(tr, 14)
    plus_di_14 = 100 * wilders_smooth(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilders_smooth(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smooth(dx, 14)
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX filter: trend strength > 25
        adx_strong = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Alligator uptrend + strong ADX + volume confirmation
            if alligator_up and adx_strong and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + strong ADX + volume confirmation
            elif alligator_down and adx_strong and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator turns down OR ADX weakens
            exit_condition = not alligator_up or adx_aligned[i] <= 25
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator turns up OR ADX weakens
            exit_condition = not alligator_down or adx_aligned[i] <= 25
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals