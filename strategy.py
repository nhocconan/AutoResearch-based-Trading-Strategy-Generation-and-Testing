#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_V1
Hypothesis: 1h Camarilla R1/S1 breakout with volume confirmation and 4h/1d HTF trend filter. Uses 4h EMA34 and 1d EMA50 for trend alignment. Targets 15-35 trades/year by requiring confluence of HTF trend, volume spike, and precise Camarilla level breaks. Works in bull/bear via HTF trend filter and mean reversion in choppy markets (CHOP > 61.8). Discrete sizing 0.20 minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for EMA34, 1d for EMA50 and CHOP)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d_arr, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d_arr, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_sum_1d = tr_1d.rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    range_1d = highest_high_1d - lowest_low_1d
    chop_1d = 100 * np.log10(chop_sum_1d / np.maximum(range_1d, 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1h Camarilla Pivot Points (using previous day OHLC) ===
    # For 1h timeframe, we need daily OHLC to calculate Camarilla levels
    # We'll use the 1d data and align it to 1h
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla equations:
    # H = high, L = low, C = close of previous day
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # R2 = C + (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # S2 = C - (H-L)*1.1/6
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    
    H = high_1d
    L = low_1d
    C = close_1d
    
    R1 = C + (H - L) * 1.1 / 12
    R2 = C + (H - L) * 1.1 / 6
    R3 = C + (H - L) * 1.1 / 4
    R4 = C + (H - L) * 1.1 / 2
    S1 = C - (H - L) * 1.1 / 12
    S2 = C - (H - L) * 1.1 / 6
    S3 = C - (H - L) * 1.1 / 4
    S4 = C - (H - L) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # === 1h Volume Confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) 
            or np.isnan(chop_1d_aligned[i]) or np.isnan(R1_aligned[i]) 
            or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        vol_ok = volume > 1.5 * vol_ma[i]  # volume spike confirmation
        
        # HTF trend alignment: both 4h and 1d EMAs agree
        uptrend = ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
        downtrend = ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (good for mean reversion at S1/R1)
        is_choppy = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume and HTF uptrend OR choppy market (mean reversion)
            if price > R1_aligned[i] and vol_ok and (uptrend or is_choppy):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume and HTF downtrend OR choppy market (mean reversion)
            elif price < S1_aligned[i] and vol_ok and (downtrend or is_choppy):
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: break below S1 or loss of volume/momentum
            if price < S1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit conditions: break above R1 or loss of volume/momentum
            if price > R1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_HTFTrend_V1"
timeframe = "1h"
leverage = 1.0