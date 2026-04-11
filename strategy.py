# 4h_1d_camarilla_trend_volume_v1
# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume spike
# Uses 1d ADX to filter for trending markets only (avoid chop)
# Long at L3 support in uptrend, short at H3 resistance in downtrend
# Volume spike confirms institutional interest at pivot levels
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag
# Works in bull/bear by only trading with 1d trend direction
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14 = adx  # Already smoothed
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 1d average volume for spike detection (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Typical price = (H+L+C)/3
    typical_price = (high_1d + low_1d + close_1d) / 3
    # Camarilla multipliers
    camarilla_mult = [1.1/12, 1.1/6, 1.1/4, 1.1/2]  # L3,L2,L1,H1,H2,H3
    
    # Calculate levels for each day
    camarilla_levels = []
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_levels.append([np.nan]*8)  # Not enough data
            continue
        # Use previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        pv = (ph + pl + pc) / 3
        rang = ph - pl
        
        levels = [
            pc - rang * camarilla_mult[0],  # L3
            pc - rang * camarilla_mult[1],  # L2
            pc - rang * camarilla_mult[2],  # L1
            pc + rang * camarilla_mult[2],  # H1
            pc + rang * camarilla_mult[1],  # H2
            pc + rang * camarilla_mult[0],  # H3
            pv,                             # Pivot
            rang                            # Range
        ]
        camarilla_levels.append(levels)
    
    camarilla_array = np.array(camarilla_levels)
    l3 = camarilla_array[:, 0]
    l2 = camarilla_array[:, 1]
    l1 = camarilla_array[:, 2]
    h1 = camarilla_array[:, 3]
    h2 = camarilla_array[:, 4]
    h3 = camarilla_array[:, 5]
    pivot = camarilla_array[:, 6]
    
    # Align all levels to 4h timeframe
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 30 to ensure sufficient data
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_14_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_spike = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]  # 50% above average
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_14_aligned[i] > 25
        
        price = close[i]
        
        # Long at L3 support in uptrend with volume spike
        long_setup = price <= l3_aligned[i] * 1.001  # Small buffer for slippage
        long_trend = price > pivot_aligned[i]  # Above pivot = uptrend bias
        long_signal = long_setup and long_trend and vol_spike and trending
        
        # Short at H3 resistance in downtrend with volume spike
        short_setup = price >= h3_aligned[i] * 0.999  # Small buffer for slippage
        short_trend = price < pivot_aligned[i]  # Below pivot = downtrend bias
        short_signal = short_setup and short_trend and vol_spike and trending
        
        # Exit when price returns to pivot level
        exit_long = price >= pivot_aligned[i] * 0.999
        exit_short = price <= pivot_aligned[i] * 1.001
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals