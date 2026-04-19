#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1_S1_Breakout_Volume_Trend_v1"
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
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d pivot levels (R1, S1) from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    
    # Align 1d pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d ATR for volatility filter and stoploss
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or \
           np.isnan(s1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(ema34_12h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below EMA34
        above_ema = price > ema34_12h[i]
        below_ema = price < ema34_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and trend alignment
            if price > r1_1d_aligned[i] and volume_ok and above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trend alignment
            elif price < s1_1d_aligned[i] and volume_ok and below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Hold long: maintain position
            signals[i] = 0.25
            # Exit: price drops below S1 or trend turns bearish
            if price < s1_1d_aligned[i] or price < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Hold short: maintain position
            signals[i] = -0.25
            # Exit: price rises above R1 or trend turns bullish
            if price > r1_1d_aligned[i] or price > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
    
    return signals