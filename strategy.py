#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_1dTrendFilter_v1
Hypothesis: 12h TRIX (12,20) crossing zero with volume spike (>1.8x avg) and 1d EMA50 trend filter.
TRIX momentum oscillator identifies trend changes early; volume confirmation reduces false signals.
Long when TRIX crosses above zero with volume and price > 1d EMA50; short when TRIX crosses below zero with volume and price < 1d EMA50.
ATR(14) stoploss (2.0x) and discrete sizing (0.25). Works in both bull/bear via 1d trend alignment.
Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h TRIX (12,20) ===
    close = prices['close'].values
    # EMA1: 12-period
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA2: 12-period of EMA1
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # EMA3: 12-period of EMA2
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # TRIX: 1-period percent change of EMA3, then 20-period EMA
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3
    trix = pd.Series(trix_raw).ewm(span=20, adjust=False, min_periods=20).mean().values * 100
    
    # === 12h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (50-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        ema_trend = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        # TRIX zero-cross signals
        trix_cross_up = (trix_prev <= 0) and (trix_now > 0)
        trix_cross_down = (trix_prev >= 0) and (trix_now < 0)
        
        if position == 0:
            # Long: TRIX crosses up + volume + price above 1d EMA50 (bullish trend)
            if trix_cross_up and volume_confirmed and (price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: TRIX crosses down + volume + price below 1d EMA50 (bearish trend)
            elif trix_cross_down and volume_confirmed and (price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: 2.0x ATR
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price below 1d EMA50
            elif price < ema_trend:
                signals[i] = 0.0
                position = 0
            # TRIX cross down exit (momentum fading)
            elif trix_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: 2.0x ATR
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal: price above 1d EMA50
            elif price > ema_trend:
                signals[i] = 0.0
                position = 0
            # TRIX cross up exit (momentum fading)
            elif trix_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_1dTrendFilter_v1"
timeframe = "12h"
leverage = 1.0