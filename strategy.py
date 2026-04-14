#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot breakout with volume confirmation and Choppiness Index regime filter
# Long when price breaks above H4 resistance AND volume > 1.5x 20-period average AND Choppiness < 61.8 (trending regime)
# Short when price breaks below L4 support AND volume > 1.5x 20-period average AND Choppiness < 61.8 (trending regime)
# Exit when price crosses back inside the Camarilla H3-L3 range
# Camarilla levels provide strong intraday support/resistance levels that work well in trending markets
# Volume confirmation ensures breakouts have conviction
# Choppiness filter avoids whipsaws in ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily range for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    
    # Calculate Camarilla levels (based on previous day's range)
    # H4 = close + 1.5 * range, L4 = close - 1.5 * range (breakout levels)
    # H3 = close + 1.25 * range, L3 = close - 1.25 * range (exit levels)
    camarilla_h4 = close_1d + 1.5 * daily_range
    camarilla_l4 = close_1d - 1.5 * daily_range
    camarilla_h3 = close_1d + 1.25 * daily_range
    camarilla_l3 = close_1d - 1.25 * daily_range
    
    # Align Camarilla levels to 4h timeframe (available after daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above H4 resistance + volume confirmation + trending regime
            if (price > camarilla_h4_aligned[i] and vol > vol_threshold and chop_values[i] < 61.8):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L4 support + volume confirmation + trending regime
            elif (price < camarilla_l4_aligned[i] and vol > vol_threshold and chop_values[i] < 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L3 support
            if price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H3 resistance
            if price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Chop"
timeframe = "4h"
leverage = 1.0