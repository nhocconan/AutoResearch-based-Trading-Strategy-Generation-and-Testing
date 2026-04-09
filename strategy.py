#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (L3/H3) breakout + 1w EMA200 trend + volume confirmation
# Camarilla levels from daily provide mean-reversion structure; breakouts beyond L3/H3 indicate strong momentum
# 1w EMA200 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (1.8x 24-period avg) filters weak breakouts
# Works in bull/bear: 1w EMA200 trend filter avoids ranging market failures
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1w_camarilla_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend direction
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h4_1d = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l4_1d = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 12h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Calculate 24-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 24:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 24-period average
        volume_confirmed = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 1w EMA200 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 1w EMA200 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 1w EMA200 trend filter
            if volume_confirmed:
                # Long entry: price > Camarilla H3 AND price > 1w EMA200 (bullish breakout + uptrend)
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_200_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L3 AND price < 1w EMA200 (bearish breakout + downtrend)
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_200_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals