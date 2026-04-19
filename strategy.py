#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with daily volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 (Camarilla resistance) AND volume > 1.5x daily average volume AND ADX(14) > 20 (trending market)
# Short when price breaks below S1 (Camarilla support) AND volume > 1.5x daily average volume AND ADX(14) > 20
# Exit when price crosses back through the Camarilla midpoint (close of previous day)
# Uses Camarilla pivot levels for precise intraday levels, volume for confirmation, ADX for trend strength.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).
name = "12h_Camarilla_R1S1_Volume_ADXFilter"
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
    
    # Get daily data for Camarilla pivots, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_mid = prev_close  # using previous close as midpoint/reference
    
    # Calculate daily average volume (20-day MA)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) for trend strength
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(H-PH, 0) if H-PH > PL-L else 0
    # -DM = max(PL-L, 0) if PL-L > H-PH else 0
    dm_plus = np.where((prev_high - df_1d['high'].shift(2).values) > (df_1d['low'].shift(2).values - prev_low),
                       np.maximum(prev_high - df_1d['high'].shift(2).values, 0), 0)
    dm_minus = np.where((df_1d['low'].shift(2).values - prev_low) > (prev_high - df_1d['high'].shift(2).values),
                        np.maximum(df_1d['low'].shift(2).values - prev_low, 0), 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 14) + 1  # ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        mid = camarilla_mid_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        adx_val = adx_aligned[i]
        
        # Trend filter: only trade when ADX > 20 (trending market)
        trending = adx_val > 20
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1 and vol > 1.5 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1 and vol > 1.5 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals