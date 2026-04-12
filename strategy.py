#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w trend filter + volume confirmation
    # Uses 1w EMA200 for trend filter: only take breakouts in direction of weekly trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false breakouts
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous 1d bar (yesterday's) OHLC to calculate today's pivots
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        pivot = (ph + pl + 2 * pc) / 4
        camarilla_pivot[i] = pivot
        camarilla_h1[i] = pivot + 1.1 * (ph - pl) / 12
        camarilla_l1[i] = pivot - 1.1 * (ph - pl) / 12
        camarilla_h2[i] = pivot + 1.1 * (ph - pl) / 6
        camarilla_l2[i] = pivot - 1.1 * (ph - pl) / 6
        camarilla_h3[i] = pivot + 1.1 * (ph - pl) / 4
        camarilla_l3[i] = pivot - 1.1 * (ph - pl) / 4
        camarilla_h4[i] = pivot + 1.1 * (ph - pl) / 2
        camarilla_l4[i] = pivot - 1.1 * (ph - pl) / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above H4 level in bullish trend
        if bullish_trend:
            long_entry = (close[i] > camarilla_h4_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below L4 level in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l4_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite H4/L4 level or trend reversal
        long_exit = (bearish_trend and close[i] < camarilla_l4_aligned[i]) or (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and close[i] > camarilla_h4_aligned[i]) or (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0