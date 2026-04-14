#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot reversal with 1-day volume confirmation and choppiness filter
# Long when price touches Camarilla L3 level AND volume > 1.5x 20-period average AND daily choppiness > 61.8 (range)
# Short when price touches Camarilla H3 level AND volume > 1.5x 20-period average AND daily choppiness > 61.8 (range)
# Exit when price reaches opposite Camarilla level (L3 for shorts, H3 for longs) or reverses 50% from entry
# Uses Camarilla pivots for intraday support/resistance, volume for confirmation, choppiness to avoid trending markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels on 12h using previous 12h bar's range
    # Camarilla uses previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First values: use current
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla multipliers
    range_val = prev_high - prev_low
    camarilla_H3 = prev_close + range_val * 1.1 / 6
    camarilla_L3 = prev_close - range_val * 1.1 / 6
    camarilla_H4 = prev_close + range_val * 1.1 / 2
    camarilla_L4 = prev_close - range_val * 1.1 / 2
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate choppiness on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness: 100 * log10(sum(TR14)/(ATR14*14)) / log10(14)
    chop_value = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    
    # Align 1d data to 12h
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        chop_value = chop_aligned[i]
        
        if position == 0:
            # Long setup: price touches L3 AND volume confirmation AND choppy market (range)
            if (price <= camarilla_L3_aligned[i] * 1.001 and price >= camarilla_L3_aligned[i] * 0.999 and 
                vol > vol_threshold and chop_value > 61.8):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 AND volume confirmation AND choppy market (range)
            elif (price >= camarilla_H3_aligned[i] * 0.999 and price <= camarilla_H3_aligned[i] * 1.001 and 
                  vol > vol_threshold and chop_value > 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 (profit target) or reverses 50% from entry toward H4
            entry_price = camarilla_L3_aligned[i]  # approximate entry near L3
            mid_point = (entry_price + camarilla_H3_aligned[i]) / 2
            if price >= camarilla_H3_aligned[i] * 0.999:  # reached H3
                position = 0
                signals[i] = 0.0
            elif price <= mid_point:  # reversed 50%
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 (profit target) or reverses 50% from entry toward L3
            entry_price = camarilla_H3_aligned[i]  # approximate entry near H3
            mid_point = (entry_price + camarilla_L3_aligned[i]) / 2
            if price <= camarilla_L3_aligned[i] * 1.001:  # reached L3
                position = 0
                signals[i] = 0.0
            elif price >= mid_point:  # reversed 50%
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_1dVolume_Chop"
timeframe = "12h"
leverage = 1.0