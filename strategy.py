#!/usr/bin/env python3
# 4h_1d_4h_vwap_vwap_std_v1
# Strategy: 4h price action relative to 1-day VWAP bands with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: The 1-day VWAP acts as a dynamic fair value with bands capturing institutional interest. Price deviations beyond 1 standard deviation from VWAP, confirmed by volume spikes and aligned with the 4-hour trend, provide high-probability mean-reversion or continuation entries. Designed for low trade frequency to avoid fee drag in both trending and ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4h_vwap_vwap_std_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day VWAP and standard deviation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Calculate standard deviation of price from VWAP
    deviation = typical_price - vwap
    variance = (deviation ** 2 * df_1d['volume']).cumsum() / vwap_denominator
    vwap_std = np.sqrt(variance)
    
    # VWAP bands (1 standard deviation)
    vwap_upper = vwap + vwap_std
    vwap_lower = vwap - vwap_std
    
    # Align VWAP and bands to 4h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper.values)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower.values)
    
    # 4-hour trend filter: EMA crossover
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_aligned[i]) or np.isnan(vwap_upper_aligned[i]) or 
            np.isnan(vwap_lower_aligned[i]) or np.isnan(trend_up[i]) or 
            np.isnan(trend_down[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals
        # Price below lower VWAP band (oversold)
        oversold = close[i] < vwap_lower_aligned[i]
        # Price above upper VWAP band (overbought)
        overbought = close[i] > vwap_upper_aligned[i]
        
        # Entry conditions
        # Long: Oversold AND uptrend AND volume confirmation
        if oversold and trend_up[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Overbought AND downtrend AND volume confirmation
        elif overbought and trend_down[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price returns to VWAP (mean reversion complete)
        elif position == 1 and close[i] >= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals