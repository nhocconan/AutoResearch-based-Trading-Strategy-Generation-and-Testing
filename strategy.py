#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout filtered by 1-week EMA20 trend and volume spike (2.0x average).
In trending markets (price > EMA20_1w for long, < for short): breakout continuation (long above upper band, short below lower band).
Volume confirmation filters false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year). Works in bull/bear via 1-week trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA20 trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w OHLC for EMA20 trend ===
    df_1w_close = df_1w['close'].values
    ema_20_1w = pd.Series(df_1w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === Daily Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high over last 20 days
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 days
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-day average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) 
            or np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 2.0x average (strict filter)
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Only enter in trending markets (price > EMA20_1w for long, < for short)
            # Volume confirmation required to avoid false breakouts
            long_condition = (price > upper) and (price > ema_trend) and volume_confirmed
            short_condition = (price < lower) and (price < ema_trend) and volume_confirmed
            
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
            elif price > upper:
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
            elif price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0