#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: 1h Camarilla (R1/S1) breakout with 4h EMA50 trend filter and 1d volume spike (>2x 20-period average).
Uses 1h timeframe for entry timing, 4h for trend direction, 1d for volume regime filter.
Discrete position sizing (0.20) to limit fee drag. Designed for 15-37 trades/year on BTC/ETH.
Works in bull/bear via 4h trend filter and volume confirmation to avoid choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h OHLC for Camarilla pivot calculation (based on previous 4h bar) ===
    df_4h_open = df_4h['open'].values
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    df_4h_close = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    range_4h = df_4h_high - df_4h_low
    r1_4h = df_4h_close + 0.275 * range_4h
    s1_4h = df_4h_close - 0.275 * range_4h
    
    # Align 4h Camarilla levels to 1h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 4h EMA50 for trend filter ===
    ema_50_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Volume filter: 20-period average ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], use DatetimeIndex for .hour
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) 
            or np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = prices['volume'].values[i]
        vol_average = vol_ma_1d_aligned[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 20-period 1d average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > R1 (breakout), 4h uptrend, volume filter
            long_breakout = price > r1_4h_aligned[i]
            long_trend = price > ema_50_4h_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 4h downtrend, volume filter
            short_breakout = price < s1_4h_aligned[i]
            short_trend = price < ema_50_4h_aligned[i]
            
            # Entry logic - balanced filters for optimal trade frequency
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.20
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.0x ATR
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Stoploss: 2.0x ATR
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0