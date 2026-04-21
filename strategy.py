#!/usr/bin/env python3
"""
6h_HTF_1d_Camarilla_R3S3_Fade_Reverse_V2
Hypothesis: Use 1d Camarilla R3/S3 levels for mean reversion fades in ranging markets (ADX < 20) and breakout continuations in trending markets (ADX > 25). 
Volume confirmation (>1.5x 20-bar MA) filters false signals. Discrete sizing (0.25) balances risk and return. 
Works in bull (captures continuation) and bear (fades overextended moves during reversals). Target 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Camarilla levels
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3, S3, R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r3 = close_1d + (high_1d - low_1d) * 1.1 * 3.0 / 12.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 * 3.0 / 12.0
    r4 = close_1d + (high_1d - low_1d) * 1.1 * 4.0 / 12.0
    s4 = close_1d - (high_1d - low_1d) * 1.1 * 4.0 / 12.0
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX (14-period) for regime filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Regime-dependent logic
            if adx[i] > 25.0:  # Trending market: breakout continuation
                # Long: break above R4 with volume
                if price > r3_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S4 with volume
                elif price < s3_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            elif adx[i] < 20.0:  # Ranging market: mean reversion fade
                # Long: fade from S3 toward pivot
                if price < s3_aligned[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                # Short: fade from R3 toward pivot
                elif price > r3_aligned[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            # ADX between 20-25: no trade (transition zone)
        
        elif position == 1:
            # Exit conditions
            if adx[i] > 25.0:  # Trending: trail with ATR stop
                if price < r3_aligned[i-1] - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging or transition: target mean reversion
                if price > pivot[i-1] or price > s3_aligned[i-1] + 1.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if adx[i] > 25.0:  # Trending: trail with ATR stop
                if price > s3_aligned[i-1] + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging or transition: target mean reversion
                if price < pivot[i-1] or price < r3_aligned[i-1] - 1.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_HTF_1d_Camarilla_R3S3_Fade_Reverse_V2"
timeframe = "6h"
leverage = 1.0