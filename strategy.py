#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Target_Zone_Volume_Confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and key levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    s1_1w = (2 * pivot_1w) - high_1w
    r1_1w = (2 * pivot_1w) - low_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    r2_1w = pivot_1w + (high_1w - low_1w)
    
    # Align weekly data to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume confirmation (20-period average)
    vol_ma_d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_d > 0, vol_ma_d, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Daily volatility filter (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ratio = atr / np.where(close > 0, close, 1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 300
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price in bullish zone (above pivot) + above weekly EMA34 + volume + volatility filter
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and
                vol_ratio[i] > 1.3 and
                atr_ratio[i] < 0.08):  # Avoid extremely high volatility
                # Avoid overextension beyond R1
                if close[i] <= r1_1w_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price in bearish zone (below pivot) + below weekly EMA34 + volume + volatility filter
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and
                  vol_ratio[i] > 1.3 and
                  atr_ratio[i] < 0.08):
                # Avoid overextension beyond S1
                if close[i] >= s1_1w_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below pivot OR below weekly EMA34
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above pivot OR above weekly EMA34
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals