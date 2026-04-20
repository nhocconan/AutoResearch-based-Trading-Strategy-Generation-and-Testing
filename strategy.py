#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Pivot_Volume_Squeeze_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h: EMA20 for trend direction ===
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1d: Standard pivot points (P, R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align pivot levels
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1h: Bollinger Bands (20,2) for squeeze detection ===
    close = prices['close'].values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_width = (upper - lower) / sma_20
    
    # === 1h: Volume filter (current > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_20_4h_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        current_close = prices['close'].iloc[i]
        current_bb_width = bb_width.iloc[i]
        current_vol_ratio = vol_ratio.iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(current_bb_width) or np.isnan(current_vol_ratio)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width below 20th percentile (tight range)
        # Calculate percentile using lookback window
        if i >= 50:
            bb_width_past = bb_width.iloc[i-50:i]
            bb_width_threshold = np.percentile(bb_width_past, 20)
            squeeze = current_bb_width < bb_width_threshold
        else:
            squeeze = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above 4h EMA20 (uptrend)
            # 2. Price breaks above R1 with volume and squeeze breakout
            if (current_close > ema_trend and
                current_close > r1 and
                current_vol_ratio > 1.5 and
                squeeze):
                signals[i] = 0.20
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 4h EMA20 (downtrend)
            # 2. Price breaks below S1 with volume and squeeze breakout
            elif (current_close < ema_trend and
                  current_close < s1 and
                  current_vol_ratio > 1.5 and
                  squeeze):
                signals[i] = -0.20
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Exit conditions:
            # 1. Price falls below 4h EMA20 (trend change)
            # 2. Price hits S1 (take profit at support)
            # 3. Squeeze re-enters (low volatility - exit)
            if (current_close < ema_trend or
                current_close <= s1 or
                not squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions:
            # 1. Price rises above 4h EMA20 (trend change)
            # 2. Price hits R1 (take profit at resistance)
            # 3. Squeeze re-enters (low volatility - exit)
            if (current_close > ema_trend or
                current_close >= r1 or
                not squeeze):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals