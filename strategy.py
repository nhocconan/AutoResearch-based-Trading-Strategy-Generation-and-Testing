#!/usr/bin/env python3
# 1h_4h1d_donchian_volume_trend_v1
# Hypothesis: 1h trend following using 4h Donchian channels for direction and 1d volume filter for strength.
# Long: price > 4h Donchian upper (20) AND 1d volume > 20-period average AND 1h momentum > 0 (session 08-20 UTC)
# Short: price < 4h Donchian lower (20) AND 1d volume > 20-period average AND 1h momentum < 0 (session 08-20 UTC)
# Exit: price crosses 4h Donchian midpoint OR volume drops below average OR momentum reverses.
# Uses 4h for signal direction (low frequency), 1h only for entry timing within session.
# Volume filter ensures trades occur during strong moves, reducing whipsaws in ranging markets.
# Session filter (08-20 UTC) avoids low-liquidity periods.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_donchian_volume_trend_v1"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h momentum (3-period ROC)
    momentum = np.full(n, np.nan)
    for i in range(3, n):
        momentum[i] = (close[i] - close[i-3]) / close[i-3]
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(high_4h), np.nan)
    donchian_mid = np.full(len(high_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Align 4h Donchian to 1h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # 1d volume average (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        mom = momentum[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        mid = donchian_mid_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol_now = df_1d['volume'].values[-1] if len(df_1d) > 0 else 0  # Simplified: use current 1d volume
        
        # Get current 1d volume (simplified approach)
        # Find the most recent completed 1d bar
        if i >= 24:  # At least 24 1h bars = 1 day
            idx_1d = i // 24
            if idx_1d < len(df_1d):
                vol_now = df_1d['volume'].values[idx_1d]
            else:
                vol_now = 0
        else:
            vol_now = 0
        
        if np.isnan(mom) or np.isnan(upper) or np.isnan(lower) or np.isnan(mid) or np.isnan(vol_ma):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if mom <= 0 or close[i] < mid or vol_now < vol_ma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            if mom >= 0 or close[i] > mid or vol_now < vol_ma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if mom > 0 and close[i] > upper and vol_now > vol_ma:
                position = 1
                signals[i] = 0.20
            elif mom < 0 and close[i] < lower and vol_now > vol_ma:
                position = -1
                signals[i] = -0.20
    
    return signals