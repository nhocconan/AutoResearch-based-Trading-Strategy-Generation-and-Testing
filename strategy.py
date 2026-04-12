#!/usr/bin/env python3
"""
1d_1w_Camarilla_Reversal_v1
Hypothesis: On daily timeframe, trade reversals at weekly Camarilla levels (H3/L3) with volume confirmation and volatility regime filter.
Goes long when price breaks above weekly H3 with above-average volume in low volatility regime.
Goes short when price breaks below weekly L3 with above-average volume in low volatility regime.
Exits when price crosses the weekly H4/L4 levels (mean reversion).
Designed for low trade frequency (10-25/year) by requiring multiple confluence factors.
Works in both bull and bear markets via volatility regime filter that avoids choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Reversal_v1"
timeframe = "1d"
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
    
    # Weekly Camarilla pivot levels (using previous week's close)
    close_prev = np.concatenate([[close_1w[0]], close_1w[:-1]])  # previous week's close
    range_1w = high_1w - low_1w
    
    h3 = close_prev + (range_1w * 1.1 / 4)
    l3 = close_prev - (range_1w * 1.1 / 4)
    h4 = close_prev + (range_1w * 1.1)
    l4 = close_prev - (range_1w * 1.1)
    
    # === DAILY VOLATILITY REGIME FILTER ===
    # Daily ATR(10) for volatility measurement
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = np.full_like(tr, np.nan, dtype=np.float64)
    for i in range(len(tr)):
        if i < 10:
            continue
        elif i == 10:
            atr_10[i] = np.nanmean(tr[1:i+1])
        else:
            atr_10[i] = (atr_10[i-1] * 9 + tr[i]) / 10
    
    # Volatility regime: low volatility = trending market
    vol_ma = np.full_like(atr_10, np.nan, dtype=np.float64)
    for i in range(len(atr_10)):
        if i < 20:
            continue
        else:
            vol_ma[i] = np.mean(atr_10[i-19:i+1])
    # Low volatility regime (trending) when current ATR < MA
    vol_regime = atr_10 < vol_ma
    
    # Align weekly data to daily timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Volume average (20-period for confirmation)
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
    
    for i in range(30, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
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