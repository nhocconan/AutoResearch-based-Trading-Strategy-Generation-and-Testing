#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v2
Hypothesis: 1d Camarilla R1/S1 breakouts filtered by 1w EMA34 trend and volume spike (>2x average).
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. ATR-based stoploss with 1.5x ATR.
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year). Works in bull/bear via 1w trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels (based on previous day's range)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_multiplier = 1.1 / 12
    
    # Shift by 1 to use previous day's OHLC (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar has no previous
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_multiplier * camarilla_range
    s1 = prev_close - camarilla_multiplier * camarilla_range
    
    # === 1w EMA34 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(35, n):  # warmup for EMA34 and previous day data
        # Skip if indicators not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_34_1w_aligned[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 2x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > 1d R1, 1w uptrend, volume spike
            long_breakout = price > r1[i]
            long_trend = price > ema_34_1w_aligned[i]
            
            # Short conditions: price < 1d S1, 1w downtrend, volume spike
            short_breakout = price < s1[i]
            short_trend = price < ema_34_1w_aligned[i]
            
            # Entry logic - ONLY enter on volume spike + trend alignment
            if long_breakout and long_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1d S1 (support broken)
            elif price < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1d R1 (resistance broken)
            elif price > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike_ATRStop_v2"
timeframe = "1d"
leverage = 1.0