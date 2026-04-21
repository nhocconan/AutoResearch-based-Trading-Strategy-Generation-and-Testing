#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_EMA34
Hypothesis: 6h Camarilla pivot fade at R3/S3 levels with 1d trend filter (EMA34) and volume confirmation (1.5x average).
In uptrend (price > EMA34), short at R3 with stop at R4; in downtrend (price < EMA34), long at S3 with stop at S4.
Volume confirmation reduces false signals. Designed for mean reversion in ranging markets with trend alignment.
Uses 6h primary and 1d HTF for Camarilla pivot and trend. Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and EMA34 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r3_1d = df_1d_close + 1.1 * range_1d
    s3_1d = df_1d_close - 1.1 * range_1d
    r4_1d = df_1d_close + 1.382 * range_1d
    s4_1d = df_1d_close - 1.382 * range_1d
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h EMA20 for dynamic stop (optional refinement) ===
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for volatility check ===
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Mean reversion entries with trend filter and volume confirmation
            # In uptrend: short at R3 (fade the breakout)
            # In downtrend: long at S3 (fade the breakdown)
            short_condition = (price > r3) and (price > ema_trend) and volume_confirmed
            long_condition = (price < s3) and (price < ema_trend) and volume_confirmed
            
            if short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
            elif long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
        
        elif position == 1:
            # Long position: exit conditions
            # Stoploss at S4 (below S3)
            if price < s4:
                signals[i] = 0.0
                position = 0
            # Take profit at midpoint between S3 and close (mean reversion target)
            elif price > (s3 + close[i]) / 2:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit conditions
            # Stoploss at R4 (above R3)
            if price > r4:
                signals[i] = 0.0
                position = 0
            # Take profit at midpoint between R3 and close (mean reversion target)
            elif price < (r3 + close[i]) / 2:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_EMA34"
timeframe = "6h"
leverage = 1.0