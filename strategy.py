#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d Kumo twist and volume confirmation.
# Long when Tenkan > Kijun, price above Kumo, Kumo bullish (Senkou A > Senkou B), and volume > 1.5x 20-period average.
# Short when Tenkan < Kijun, price below Kumo, Kumo bearish (Senkou A < Senkou B), and volume > 1.5x 20-period average.
# Exit when Tenkan/Kijun cross reverses or price crosses Kumo in opposite direction.
# Uses Kumo twist from higher timeframe to filter false signals in ranging markets.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_Ichimoku_1dKumoTwist_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (9, 26, 52 periods)
    conversion_line = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                       pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    base_line = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    leading_span_a = ((conversion_line + base_line) / 2).shift(26)
    leading_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                       pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    conversion_line_aligned = align_htf_to_ltf(prices, df_1d, conversion_line.values)
    base_line_aligned = align_htf_to_ltf(prices, df_1d, base_line.values)
    leading_span_a_aligned = align_htf_to_ltf(prices, df_1d, leading_span_a.values)
    leading_span_b_aligned = align_htf_to_ltf(prices, df_1d, leading_span_b.values)
    
    # 6h volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(conversion_line_aligned[i]) or np.isnan(base_line_aligned[i]) or
            np.isnan(leading_span_a_aligned[i]) or np.isnan(leading_span_b_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Kumo twist: Senkou A > Senkou B = bullish, Senkou A < Senkou B = bearish
        kumo_bullish = leading_span_a_aligned[i] > leading_span_b_aligned[i]
        kumo_bearish = leading_span_a_aligned[i] < leading_span_b_aligned[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > max(leading_span_a_aligned[i], leading_span_b_aligned[i])
        price_below_kumo = close[i] < min(leading_span_a_aligned[i], leading_span_b_aligned[i])
        
        if position == 0:
            # Long conditions: Bullish TK cross, price above Kumo, bullish Kumo, volume
            tk_bullish = conversion_line_aligned[i] > base_line_aligned[i]
            tk_bullish_prev = conversion_line_aligned[i-1] <= base_line_aligned[i-1]
            long_cond = tk_bullish and tk_bullish_prev and price_above_kumo and kumo_bullish and volume_filter[i]
            
            # Short conditions: Bearish TK cross, price below Kumo, bearish Kumo, volume
            tk_bearish = conversion_line_aligned[i] < base_line_aligned[i]
            tk_bearish_prev = conversion_line_aligned[i-1] >= base_line_aligned[i-1]
            short_cond = tk_bearish and tk_bearish_prev and price_below_kumo and kumo_bearish and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross turns bearish OR price drops below Kumo
            tk_bearish = conversion_line_aligned[i] < base_line_aligned[i]
            price_below_kumo = close[i] < min(leading_span_a_aligned[i], leading_span_b_aligned[i])
            if tk_bearish or price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross turns bullish OR price rises above Kumo
            tk_bullish = conversion_line_aligned[i] > base_line_aligned[i]
            price_above_kumo = close[i] > max(leading_span_a_aligned[i], leading_span_b_aligned[i])
            if tk_bullish or price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals