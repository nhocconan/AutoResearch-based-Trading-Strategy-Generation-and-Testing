#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h Camarilla pivot reversals with daily volume spike and 1d chop regime filter.
# Long: price touches L3/L4 support, 12h volume > 1.5x 20-period avg, 1d chop > 61.8 (range)
# Short: price touches H3/H4 resistance, same filters
# Exit: price reaches opposite H/L3 level or chop < 38.2 (trend)
# Works in bull/bear by fading extremes in ranging markets (chop filter avoids trending whipsaws).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume confirmation: > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if vol_count > 20:
            vol_sum -= volume[i - 20]
            vol_count -= 1
        if vol_count >= 20:
            vol_ma[i] = vol_sum / 20.0
    vol_ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    # Daily HTF data for Camarilla pivots and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots for each day (using previous day's OHLC)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's values
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        range_val = phigh - plow
        
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily chop regime (Ehler's Chop Index) - needs 14 periods
    chop = np.full(len(close_1d), np.nan)
    atr_14 = np.full(len(close_1d), np.nan)
    atr_sum = 0.0
    atr_count = 0
    
    # Calculate ATR first
    true_range = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            true_range[i] = high_1d[i] - low_1d[i]
        else:
            true_range[i] = max(high_1d[i] - low_1d[i],
                               abs(high_1d[i] - close_1d[i-1]),
                               abs(low_1d[i] - close_1d[i-1]))
        
        atr_sum += true_range[i]
        atr_count += 1
        if atr_count > 14:
            atr_sum -= true_range[i - 14]
            atr_count -= 1
        if atr_count >= 14:
            atr_14[i] = atr_sum / 14.0
    
    # Calculate Chop: 100 * log10(sum(ATR14) / (max(high)-min(low)) * sqrt(period))
    for i in range(14, len(close_1d)):
        if not np.isnan(atr_14[i]):
            period_high = np.max(high_1d[i-13:i+1])
            period_low = np.min(low_1d[i-13:i+1])
            if period_high > period_low:
                sum_atr = atr_14[i] * 14  # approximate sum
                chop[i] = 100 * np.log10(sum_atr / (period_high - period_low) * np.sqrt(14))
    
    # Align chop to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ratio[i]) or np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or
            np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or np.isnan(chop_12h[i])):
            if position != 0:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_ratio[i] > 1.5
        chop_ok = chop_12h[i] > 61.8  # Range regime
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or chop turns trending (< 38.2)
            if price >= h3_12h[i] or chop_12h[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 or chop turns trending
            if price <= l3_12h[i] or chop_12h[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L3/L4 support in range regime
            if vol_ok and chop_ok and (price <= l3_12h[i] or price <= l4_12h[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H3/H4 resistance in range regime
            elif vol_ok and chop_ok and (price >= h3_12h[i] or price >= h4_12h[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals