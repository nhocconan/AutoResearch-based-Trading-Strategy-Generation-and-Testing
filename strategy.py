#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter_v1
Hypothesis: Daily Donchian(20) breakout filtered by weekly EMA34 trend and volume spike (2.0x average).
Designed to work in both bull and bear markets via weekly trend alignment and strict entry filters.
Uses discrete position sizing (0.25) and ATR(14) stoploss (2.0x) to manage risk and minimize fee drag.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === Daily Donchian(20) channels (based on previous 20 daily bars) ===
    # We need daily OHLC for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Donchian channels for each daily bar (using previous 20 bars)
    high_20 = pd.Series(df_1d_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d_low).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 1d timeframe (already aligned, but using helper for consistency)
    donch_high_1d = align_htf_to_ltf(prices, df_1d, high_20)
    donch_low_1d = align_htf_to_ltf(prices, df_1d, low_20)
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
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
    
    # === Daily volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high_1d[i]) or np.isnan(donch_low_1d[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        donch_high = donch_high_1d[i]
        donch_low = donch_low_1d[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Only enter in direction of weekly trend
            long_condition = (price > donch_high) and (price > ema_trend) and volume_confirmed
            short_condition = (price < donch_low) and (price < ema_trend) and volume_confirmed
            
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
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian lower band (avoid whipsaw)
            elif price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at Donchian upper band (avoid whipsaw)
            elif price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0