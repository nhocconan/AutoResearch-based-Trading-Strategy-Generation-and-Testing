#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter, volume confirmation (>2.0x 20-bar volume MA), and ATR trailing stop (3x ATR). 
Uses discrete position sizing (0.25) to limit fee churn. Designed to work in bull via breakouts and bear via short breakdowns with trend alignment.
Target: 20-50 trades/year per symbol (<150 total 4h trades) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3.0
    
    # Camarilla levels based on previous day's OHLC
    # We need to shift by 1 to avoid look-ahead (use previous day's data)
    # Since we're on 4h timeframe, we approximate daily levels using rolling window
    # For proper Camarilla, we'd need actual daily OHLC, but we approximate with 6-period (6*4h=24h) lookback
    lookback = 6  # 6 * 4h = 24h approx
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last()
    
    # Shift to use only completed periods (avoid look-ahead)
    prev_high = roll_high.shift(1).values
    prev_low = roll_low.shift(1).values
    prev_close = roll_close.shift(1).values
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    R1 = prev_close + range_val * 1.1 / 12
    S1 = prev_close - range_val * 1.1 / 12
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + price > 1d EMA50
            if price > R1[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below S1 + volume confirmation + price < 1d EMA50
            elif price < S1[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from highest since entry
            if price < highest_since_entry - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from lowest since entry
            if price > lowest_since_entry + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0