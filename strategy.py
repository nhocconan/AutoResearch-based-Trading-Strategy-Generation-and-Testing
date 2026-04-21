#!/usr/bin/env python3
"""
12h_HTF_1d_1w_DonchianBreakout_VolumeATR_V1
Hypothesis: 12h Donchian(20) breakouts with 1d EMA50 trend filter and 1w volume spike confirmation.
Targets 12-37 trades/year (50-150 total over 4 years) by requiring confluence of:
- Price breaking 20-period Donchian channel on 12h
- 1d EMA50 trend alignment (price > EMA50 for longs, < for shorts)
- 1w volume > 1.5x 20-period volume MA (institutional participation)
ATR-based stoploss and discrete position sizing (0.25) to control drawdown and fees.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend, 1w for volume confirmation)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1w Volume MA (20-period) for spike detection ===
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for stoploss and position sizing
    tr1 = pd.Series(high_12h - low_12h).values
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1))).values
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1))).values
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) 
            or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma_1w_aligned[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + 1d uptrend
            if price > highest_20[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume confirmation + 1d downtrend
            elif price < lowest_20[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long: trail stop or exit on trend reversal
            if price < ema_50_1d_aligned[i] or price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: trail stop or exit on trend reversal
            if price > ema_50_1d_aligned[i] or price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_1w_DonchianBreakout_VolumeATR_V1"
timeframe = "12h"
leverage = 1.0