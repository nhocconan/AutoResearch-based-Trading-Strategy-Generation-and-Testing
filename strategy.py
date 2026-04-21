#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter (price > weekly EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). 
Weekly EMA34 ensures alignment with higher-timeframe trend, reducing false breakouts in choppy markets. 
Volume spike confirms institutional interest. ATR-based stoploss limits drawdown. 
Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag in BTC/ETH.
Works in bull markets via breakout momentum and in bear markets via short-side breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend, 1w for Donchian channels)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1w Donchian(20) channels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    dh_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    dl_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (wait for weekly bar to close)
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, dl_20)
    
    # === 1d Indicators (primary timeframe) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume confirmation + 1d uptrend
            if price > dh_20_aligned[i] and vol_ok and price > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly Donchian low + volume confirmation + 1d downtrend
            elif price < dl_20_aligned[i] and vol_ok and price < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss or trend reversal
            stop_price = entry_price - 2.5 * atr[i]
            if price < stop_price or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss or trend reversal
            stop_price = entry_price + 2.5 * atr[i]
            if price > stop_price or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "1d"
leverage = 1.0