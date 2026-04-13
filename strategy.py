#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels, volume confirmation, and Choppiness regime filter.
# Long: Price touches or crosses above Camarilla H4 (upper resistance) + volume > 2.0x average + Choppiness < 40 (trending).
# Short: Price touches or crosses below Camarilla L4 (lower support) + volume > 2.0x average + Choppiness < 40 (trending).
# Uses 1d for pivot structure (key institutional levels), 4h for entry timing with volume and regime confirmation.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 20-50 total trades over 4 years (5-12.5/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    camarilla_h4 = np.full(len(close_1d), np.nan)  # H4 = Close + 1.5 * (High - Low)
    camarilla_l4 = np.full(len(close_1d), np.nan)  # L4 = Close - 1.5 * (High - Low)
    for i in range(1, len(close_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        camarilla_h4[i] = close_val + 1.5 * (high_val - low_val)
        camarilla_l4[i] = close_val - 1.5 * (high_val - low_val)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Choppiness Index (14-period) for regime filter
    chop = np.full(n, np.nan)
    for i in range(14, n):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of True Range over 14 periods
        atr_sum = np.sum(tr[i-14:i])
        
        # Highest high and lowest low over 14 periods
        hh = np.max(high[i-14:i])
        ll = np.min(low[i-14:i])
        
        # Choppiness calculation
        if atr_sum > 0 and hh != ll:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50.0  # neutral when undefined
    
    # Align 1d Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        chop_val = chop[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        # Regime filter: Choppiness < 40 (trending market)
        regime_filter = chop_val < 40.0
        
        if position == 0:
            # Long: price touches/above H4 + volume confirmation + trending regime
            if (price >= h4 and 
                volume_confirm and
                regime_filter):
                position = 1
                signals[i] = position_size
            # Short: price touches/below L4 + volume confirmation + trending regime
            elif (price <= l4 and 
                  volume_confirm and
                  regime_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below L4 or chop > 50 (ranging)
            if (price < l4 or
                chop_val > 50.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above H4 or chop > 50 (ranging)
            if (price > h4 or
                chop_val > 50.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Regime"
timeframe = "4h"
leverage = 1.0