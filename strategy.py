#!/usr/bin/env python3
# 12h_camarilla_1w_trend_volume_v1
# Hypothesis: 12h strategy using weekly Camarilla pivot levels (H3/L3) from 1w timeframe for structural support/resistance,
# combined with 1d trend filter (price above/below 1d EMA200) and volume confirmation (>1.5x 20-period average).
# Weekly Camarilla provides strong intraday levels that work in both bull and bear markets as reversal/continuation points.
# 1d EMA200 filter ensures we only trade in direction of higher timeframe trend.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Camarilla pivot levels (based on previous week's range)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    pivot_point_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels: H3, L3 (strongest intraday support/resistance)
    h3_1w = pivot_point_1w + (range_1w * 1.1 / 4)
    l3_1w = pivot_point_1w - (range_1w * 1.1 / 4)
    
    # Align weekly Camarilla to 12h timeframe
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # 1d HTF data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly L3 (intraday support fails)
            if close[i] < l3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly H3 (intraday resistance fails)
            if close[i] > h3_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price touches/bounces from weekly L3 WITH 1d bullish bias (price > EMA200)
                if close[i] <= l3_1w_aligned[i] * 1.005 and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches/rejects from weekly H3 WITH 1d bearish bias (price < EMA200)
                elif close[i] >= h3_1w_aligned[i] * 0.995 and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals