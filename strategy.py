#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1w trend filter and volume confirmation
# Uses wider Camarilla levels (R4/S4) for stronger breakouts, filtered by weekly trend
# to avoid counter-trend trades. Volume spike confirms institutional interest.
# Designed for low trade frequency (<40/year) to minimize fee drag in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(40) for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar (R4/S4 levels)
    camarilla_R4 = np.full(len(df_1d), np.nan)
    camarilla_S4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        # Camarilla R4 and S4 levels (widest bands)
        camarilla_R4[i] = pc + 1.1 * rang
        camarilla_S4[i] = pc - 1.1 * rang
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data (1 bar), weekly EMA (40), daily Camarilla (1 bar), volume MA (20)
    start_idx = max(1, 40, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_40_1w_aligned[i]
        bearish_trend = price < ema_40_1w_aligned[i]
        
        if position == 0:
            # Long: break above R4 with volume and bullish weekly trend
            if price > R4_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below S4 with volume and bearish weekly trend
            elif price < S4_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to S4 (mean reversion) or weekly trend turns bearish
            if price <= S4_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to R4 (mean reversion) or weekly trend turns bullish
            if price >= R4_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0