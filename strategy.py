#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_VolumeSpike_1dTrend_v1
Hypothesis: 12h TRIX zero-cross filtered by 1d EMA50 trend and volume spike.
TRIX (triple-smoothed EMA) momentum oscillator: long when TRIX crosses above zero with rising momentum,
short when crosses below zero with falling momentum. Volume confirmation (2.0x average) filters false signals.
Trend filter: only long when price > EMA50_1d, short when price < EMA50_1d. Works in bull/bear via trend alignment.
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year). Uses discrete sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === TRIX (15-period triple-smoothed EMA) ===
    close = prices['close'].values
    # First EMA
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (ema3 - prev_ema3) / prev_ema3
    trix = np.full_like(close, np.nan, dtype=np.float64)
    trix[1:] = 100.0 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
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
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirmed = volume_now > 2.0 * vol_avg
        
        # TRIX zero-cross signals
        trix_cross_up = (trix_prev <= 0) and (trix_now > 0)
        trix_cross_down = (trix_prev >= 0) and (trix_now < 0)
        
        if position == 0:
            # Only enter in trending markets with volume confirmation
            long_condition = trix_cross_up and (price > ema_trend) and volume_confirmed
            short_condition = trix_cross_down and (price < ema_trend) and volume_confirmed
            
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
            # Momentum exit: TRIX crosses below zero
            elif trix_now < 0:
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
            # Momentum exit: TRIX crosses above zero
            elif trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_ZeroCross_VolumeSpike_1dTrend_v1"
timeframe = "12h"
leverage = 1.0