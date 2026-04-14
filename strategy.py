#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot levels from 1-day data with volume confirmation and ADX filter
# Long when price touches Camarilla L3 support with volume spike and ADX > 20
# Short when price touches Camarilla H3 resistance with volume spike and ADX > 20
# Uses Camarilla's mathematically derived support/resistance levels (high-probability reversal zones)
# Volume spike confirms institutional interest, ADX filter ensures trending environment
# Target: 20-50 trades per year to minimize fee drag while capturing high-probability reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula based on previous day's range
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pp + (range_val * 1.1 / 2)  # Resistance level 3
    l3 = pp - (range_val * 1.1 / 2)  # Support level 3
    h4 = pp + (range_val * 1.1)      # Resistance level 4
    l4 = pp - (range_val * 1.1)      # Support level 4
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate ADX on 4h data for trend strength filter
    # ADX calculation requires +DI and -DI
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Smoothed values using Wilder's smoothing (EMA-like with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: previous * (period-1)/period + current / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Volume spike detection (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long setup: price touches L3 support with volume spike and ADX > 20
            if (abs(price - l3_aligned[i]) < (high[i] - low[i]) * 0.1 and  # Within 10% of range
                volume_spike[i] and adx[i] > 20):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 resistance with volume spike and ADX > 20
            elif (abs(price - h3_aligned[i]) < (high[i] - low[i]) * 0.1 and  # Within 10% of range
                  volume_spike[i] and adx[i] > 20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 resistance or closes below L3
            if price >= h3_aligned[i] or price < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 support or closes above H3
            if price <= l3_aligned[i] or price > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0