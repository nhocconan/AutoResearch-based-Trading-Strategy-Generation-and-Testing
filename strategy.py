#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_DonchianBreakout_VolumeSpike
Hypothesis: Daily KAMA trend filter with Donchian(20) breakout and volume spike (>2.0x 20-period MA) for entries. Uses 1w EMA34 for higher timeframe trend confirmation. Designed for low trade frequency (~10-20/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets via multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA34 trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === Daily OHLC for KAMA and Donchian channels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # === Daily KAMA (Efficiency Ratio = 10, Fast = 2, Slow = 30) ===
    close_1d = pd.Series(df_1d_close)
    change = abs(close_1d.diff(10)).values
    volatility = abs(close_1d.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(df_1d_close)
    kama[0] = df_1d_close[0]
    for i in range(1, len(df_1d_close)):
        kama[i] = kama[i-1] + sc[i] * (df_1d_close[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === Daily Donchian(20) channels ===
    high_20 = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === Weekly EMA34 for trend filter ===
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        kama_val = kama_aligned[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_34 = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Enter only with volume spike and multi-timeframe trend alignment
            long_condition = (price > donchian_high) and (price > kama_val) and (price > ema_34) and volume_spike
            short_condition = (price < donchian_low) and (price < kama_val) and (price < ema_34) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below KAMA or weekly EMA)
            elif price < kama_val or price < ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above KAMA or weekly EMA)
            elif price > kama_val or price > ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_DonchianBreakout_VolumeSpike"
timeframe = "1d"
leverage = 1.0