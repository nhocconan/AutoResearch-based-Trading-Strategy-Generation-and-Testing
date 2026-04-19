#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with daily volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 (38.2% retracement) with volume > 1.3x daily avg and ADX > 25 (trending market)
# Short when price breaks below S1 (61.8% retracement) with volume > 1.3x daily avg and ADX > 25
# Exit when price returns to the pivot point (central level)
# Uses Camarilla for intraday support/resistance, volume for breakout confirmation, ADX to avoid ranging markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Camarilla_Pivot_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, Pivot = (H+L+C)/3
    # Using previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    range_hl = prev_high - prev_low
    r1 = prev_close + range_hl * 1.1 / 12
    s1 = prev_close - range_hl * 1.1 / 12
    pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily average volume (20-period) for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ADX (14-period) for trend strength
    # TR = max(H-L, |H-PC|, |L-PC|)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM = max(H-PH, 0) if H-PH > PL-L else 0
    # -DM = max(PL-L, 0) if PL-L > H-PH else 0
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 100)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        adx_val = adx_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1_val and vol > 1.3 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1_val and vol > 1.3 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if price <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if price >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals