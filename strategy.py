#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1
Hypothesis: Camarilla R1/S1 breakouts on 4h filtered by 1d EMA50 trend for directional bias and volume spike (>2x 20-period average) for confirmation.
Only take longs when price > 1d EMA50 (bullish bias) and shorts when price < 1d EMA50 (bearish bias).
ATR-based stoploss (2.0x ATR) and exit when price closes back inside the Camarilla H-L range.
Designed for 20-50 trades/year per symbol (~80-200 total over 4 years) to minimize fee drag.
Uses 1d trend as regime filter to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h OHLC for Camarilla calculation (using previous day's data) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate previous day's OHLC for Camarilla levels
    # We need to group by day to get prior day's H/L/C
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    # Get prior day's high, low, close for each bar
    prev_high = df.groupby('date')['high'].shift(1).values
    prev_low = df.groupby('date')['low'].shift(1).values
    prev_close = df.groupby('date')['close'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: current volume > 2.0x 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: price > Camarilla R1, 1d uptrend, volume filter
            long_breakout = price > camarilla_r1[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < Camarilla S1, 1d downtrend, volume filter
            short_breakout = price < camarilla_s1[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes back inside Camarilla H-L range (mean reversion)
            elif camarilla_s1[i] <= price <= camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price closes back inside Camarilla H-L range (mean reversion)
            elif camarilla_s1[i] <= price <= camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0