#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (L3/H3) breakout + 12h EMA50 trend + volume confirmation
# Camarilla levels provide mean-reversion structure; breakouts beyond L3/H3 indicate strong momentum
# 12h EMA50 ensures we trade with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (1.5x 20-period avg) filters weak breakouts
# Works in bull/bear: EMA50 trend filter avoids ranging market failures
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_12h_camarilla_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        # Get previous 1d bar's OHLC (need to map 4h index to 1d index)
        # Since we're using 4h timeframe, we need the 1d bar that completed before current 4h bar
        # We'll use align_htf_to_ltf later, but for calculation we need raw 1d values
        pass  # Will calculate after getting 1d arrays
    
    # Calculate Camarilla levels from 1d OHLC
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla levels: based on previous day's range
        camarilla_h4_1d = close_1d + 1.1 * (high_1d - low_1d)
        camarilla_l4_1d = close_1d - 1.1 * (high_1d - low_1d)
        camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
        camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
        
        # Align to 4h timeframe (completed 1d bar only)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    else:
        camarilla_h3_aligned = camarilla_l3_aligned = camarilla_h4_aligned = camarilla_l4_aligned = np.full(n, np.nan)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 12h EMA50 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 12h EMA50 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + EMA50 trend filter
            if volume_confirmed:
                # Long entry: price > Camarilla H3 AND price > 12h EMA50 (bullish breakout + uptrend)
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Camarilla L3 AND price < 12h EMA50 (bearish breakout + downtrend)
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals