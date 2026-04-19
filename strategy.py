#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and 1d ADX trend filter.
# Long when price breaks above R1 and volume > 1.3x daily average volume and ADX > 25 (trending market)
# Short when price breaks below S1 and volume > 1.3x daily average volume and ADX > 25
# Exit when price crosses back through the daily pivot point (PP)
# Uses Camarilla pivots for precise intraday levels, volume for confirmation, ADX to avoid chop.
# Target: 20-30 trades/year per symbol.

name = "4h_Camarilla_Volume_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1, PP) from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    price_range = df_1d['high'] - df_1d['low']
    pp = typical_price.shift(1)  # Previous day's pivot
    r1 = pp + price_range.shift(1) * 1.1 / 12
    s1 = pp - price_range.shift(1) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d ADX for trend filter
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(H-Hprev, 0) if H-Hprev > Lprev-L else 0
    dm_plus = np.where(
        (df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']),
        np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0),
        0
    )
    # -DM = max(Lprev-L, 0) if Lprev-L > H-Hprev else 0
    dm_minus = np.where(
        (df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)),
        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0),
        0
    )
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        adx_val = adx_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
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
            # Long exit: price crosses below pivot point
            if price < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals