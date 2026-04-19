#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Pivot Breakout with Volume and 4h Trend Filter
# Uses daily pivots for structure, 4h EMA34 for trend filter, and volume confirmation
# Designed to work in both bull and bear markets by filtering trades with higher timeframe trend
# Target: 15-30 trades/year to avoid fee drag
name = "1h_4h_1d_Pivot_R1S1_Volume_EMA34Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Pivot points (structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Pivot, R1, S1
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align daily pivot levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h close
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume filter: current volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) \
           or np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Trend filter: price > EMA34 for long, price < EMA34 for short
        uptrend = price > ema_34_4h_aligned[i]
        downtrend = price < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume and downtrend
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns below S1 (mean reversion to opposite level)
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns above R1 (mean reversion to opposite level)
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals