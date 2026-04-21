#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter_v1
Hypothesis: Daily Donchian(20) breakout filtered by weekly EMA20 trend and volume spike (1.8x average).
Long when price > upper Donchian(20) and above weekly EMA20; short when price < lower Donchian(20) and below weekly EMA20.
Volume confirmation reduces false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed to capture medium-term trends in both bull and bear markets via weekly trend alignment.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Daily Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high of last 20 days (including current)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days (including current)
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly EMA20 for trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) 
            or np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper_band = upper[i]
        lower_band = lower[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Only enter in direction of weekly trend
            long_condition = (price > upper_band) and (price > ema_trend) and volume_confirmed
            short_condition = (price < lower_band) and (price < ema_trend) and volume_confirmed
            
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
            # Mean reversion exit at upper band (extreme overbought)
            elif price > upper_band * 1.02:  # 2% above upper band
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
            # Mean reversion exit at lower band (extreme oversold)
            elif price < lower_band * 0.98:  # 2% below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0