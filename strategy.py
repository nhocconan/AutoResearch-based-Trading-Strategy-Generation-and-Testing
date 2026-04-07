#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
# Hypothesis: Camarilla pivot levels act as strong support/resistance on 1d charts.
# At 6h timeframe, we look for reversals at S3/R3 levels when price is extended from 1d VWAP,
# but only in the direction of the 1d EMA50 trend to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for 60-120 trades/year.

name = "6h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    range_1d = high_1d - low_1d
    camarilla_H5 = close_1d + range_1d * 1.1 / 2
    camarilla_H4 = close_1d + range_1d * 1.1
    camarilla_H3 = close_1d + range_1d * 1.1 / 4
    camarilla_L3 = close_1d - range_1d * 1.1 / 4
    camarilla_L4 = close_1d - range_1d * 1.1
    camarilla_L5 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    H3_6h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_6h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    H4_6h = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    L4_6h = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6-period SMA for volume average (more responsive on 6h)
    vol_sma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(6, n):  # Start after warmup for volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(H3_6h[i]) or 
            np.isnan(L3_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 OR trend turns down
            if close[i] < L3_6h[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above H3 OR trend turns up
            if close[i] > H3_6h[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price crosses above L3 (support hold) + volume confirmation + uptrend
            if (close[i] > L3_6h[i] and 
                close[i-1] <= L3_6h[i-1] and  # crossed above L3
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price crosses below H3 (resistance hold) + volume confirmation + downtrend
            elif (close[i] < H3_6h[i] and 
                  close[i-1] >= H3_6h[i-1] and  # crossed below H3
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals