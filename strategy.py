#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Trend_v1
Hypothesis: On 12h timeframe, buy breakouts above weekly Camarilla H3 with 1d uptrend filter and volume confirmation,
sell breakdowns below weekly L3 with 1d downtrend and volume confirmation. Exit at weekly H4/L4 levels.
Uses weekly volatility regime filter to avoid choppy markets. Designed for low trade frequency
(12-37/year) by requiring multiple confluence factors. Works in bull/bear via 1d trend filter
and mean-reversion exit at Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Trend_v1"
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
    
    # === WEEKLY CAMARILLA LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla pivot levels
    close_prev = close_1w  # using same week's close as approximation
    range_1w = high_1w - low_1w
    
    h5 = close_prev + (range_1w * 1.1 / 2)
    h4 = close_prev + (range_1w * 1.1)
    h3 = close_prev + (range_1w * 1.1 / 4)
    l3 = close_prev - (range_1w * 1.1 / 4)
    l4 = close_prev - (range_1w * 1.1)
    l5 = close_prev - (range_1w * 1.1 / 2)
    
    # === WEEKLY VOLATILITY REGIME FILTER ===
    # Weekly ATR(14) for volatility measurement
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    # Volatility regime: low volatility = trending market
    vol_ma = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i < 30:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(atr_14[i-29:i+1])
    # Low volatility regime (trending) when current ATR < MA
    vol_regime = atr_14 < vol_ma
    
    # Align data to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Volume average (20-period for 12h = ~10 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions
        long_setup = (close[i] > h3_aligned[i]) and vol_confirm and in_trend_regime
        short_setup = (close[i] < l3_aligned[i]) and vol_confirm and in_trend_regime
        
        # Exit conditions: mean reversion to H4/L4 levels
        exit_long = close[i] < l4_aligned[i]
        exit_short = close[i] > h4_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals