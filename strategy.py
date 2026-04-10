#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d VWAP (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND price < 1d VWAP (bearish regime)
# - Exit when power signals weaken or price crosses 1d VWAP oppositely
# - Uses 1d VWAP as regime filter to align with higher timeframe trend
# - Targets 12-30 trades/year (50-120 total over 4 years) to avoid fee drag
# - Elder Ray measures bull/bear strength; VWAP filters for regime alignment
# - Works in both bull (long bias) and bear (short bias) markets via regime filter

name = "6h_1d_elder_ray_vwap_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute Elder Ray components (13-period EMA)
    ema13 = prices['close'].ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13
    bear_power = ema13 - prices['low'].values
    
    # Pre-compute 1d VWAP for regime filter
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND price > 1d VWAP
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                prices['close'].iloc[i] > vwap_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 AND Bull Power < 0 AND price < 1d VWAP
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  prices['close'].iloc[i] < vwap_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Power signals weaken (contrary pressure appears)
            # 2. Price crosses 1d VWAP in opposite direction (regime change)
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                    prices['close'].iloc[i] < vwap_1d_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                    prices['close'].iloc[i] > vwap_1d_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals