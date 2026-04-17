#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA34 > EMA89 (uptrend) AND volume > 1.5x average.
Short when price breaks below Camarilla S3 AND 12h EMA34 < EMA89 (downtrend) AND volume > 1.5x average.
Exit when price reverts to Camarilla H3/L3 level OR 12h EMA crossover reverses.
Uses 6h for price action/volume, 12h for EMA trend filter to avoid whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday pivot points,
EMA filter ensures we trade with the higher timeframe trend, volume confirmation reduces fakeouts.
Works in bull markets (captures uptrend breakouts) and bear markets (captures downtrend breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivot calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Camarilla pivot levels for 6h timeframe (based on previous 6h bar)
    # Camarilla formulas:
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    # Pivot = (high + low + close) / 3
    
    # Calculate for previous bar (to avoid look-ahead)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    
    # First bar handling
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    # Calculate Camarilla levels
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    camarilla_h2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    camarilla_l2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    camarilla_h1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_l1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume_6h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs (34 and 89)
    close_12h_series = pd.Series(close_12h)
    ema_34 = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = close_12h_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align all 6h data to lower timeframe (prices)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pivot)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    # Align 12h EMA data to lower timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_12h, ema_89)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h2 = camarilla_h2_aligned[i]
        l2 = camarilla_l2_aligned[i]
        h1 = camarilla_h1_aligned[i]
        l1 = camarilla_l1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        vol_ma = volume_ma_aligned[i]
        ema_fast = ema_34_aligned[i]
        ema_slow = ema_89_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla H3 AND 12h EMA34 > EMA89 (uptrend) AND volume > 1.5x avg
            if price > h3 and ema_fast > ema_slow and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla L3 AND 12h EMA34 < EMA89 (downtrend) AND volume > 1.5x avg
            elif price < l3 and ema_fast < ema_slow and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla H2 OR 12h EMA crossover reverses
            if price < h2 or ema_fast < ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla L2 OR 12h EMA crossover reverses
            if price > l2 or ema_fast > ema_slow:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_12hEMA_Volume_Filter"
timeframe = "6h"
leverage = 1.0