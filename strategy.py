#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter and volume confirmation.
- Primary timeframe: 6h for breakout entries.
- HTF: 1w Camarilla pivot levels (H3/L3) for regime - long only when price > weekly H3, short only when price < weekly L3.
- Volume: Current 6h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND price > weekly H3 AND volume spike.
         Short when price breaks below Donchian(20) low AND price < weekly L3 AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume/regime confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Weekly Camarilla pivots provide strong structural support/resistance that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels on 6h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA on 6h
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (H3, L3)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    typical_price_vals = typical_price.values
    
    # Weekly range
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_range = weekly_high - weekly_low
    
    # Camarilla levels
    # H3 = typical_price + (range * 1.1 / 4)
    # L3 = typical_price - (range * 1.1 / 4)
    camarilla_h3 = typical_price_vals + (weekly_range * 1.1 / 4)
    camarilla_l3 = typical_price_vals - (weekly_range * 1.1 / 4)
    
    # Align HTF indicators to 6h
    period20_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), period20_high)
    period20_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), period20_low)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, prices, vol_ma_6h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_spike = volume > (1.5 * vol_ma_6h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 5)  # Need enough bars for Donchian and weekly data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high_aligned[i]) or np.isnan(period20_low_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        donchian_high = period20_high_aligned[i]
        donchian_low = period20_low_aligned[i]
        weekly_h3 = camarilla_h3_aligned[i]
        weekly_l3 = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and regime filter
            if volume_spike[i]:
                # Bullish: price breaks above Donchian high AND price > weekly H3
                if curr_high > donchian_high and curr_close > weekly_h3:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND price < weekly L3
                elif curr_low < donchian_low and curr_close < weekly_l3:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume/regime
            if curr_low < donchian_low or not volume_spike[i] or curr_close <= weekly_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume/regime
            if curr_high > donchian_high or not volume_spike[i] or curr_close >= weekly_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyCamarilla_H3L3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0