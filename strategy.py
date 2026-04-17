#!/usr/bin/env python3
"""
4h_Volume_Spike_Reversion_v1
Buying volume spikes during mean reversion in weak trends.
- Long: Volume spike > 2.0x 30-period average AND price < BB(20,2.5) lower band AND ADX < 25
- Short: Volume spike > 2.0x 30-period average AND price > BB(20,2.5) upper band AND ADX < 25
- Exit: ADX > 30 (trend strength) OR price reverts to VWAP(20)
Uses 1d trend filter: only trade long when price > 1d EMA50, short when price < 1d EMA50.
Designed to capture mean reversion bursts during low volatility periods.
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Bollinger Bands (20, 2.5) ===
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2.5 * std20
    bb_lower = sma20 - 2.5 * std20
    
    # === Volume spike detection ===
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > 2.0 * vol_ma30
    
    # === ADX(14) for trend strength ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === VWAP(20) for exit ===
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # === 1d EMA50 for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or 
            np.isnan(vol_ma30[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: volume spike + price below BB lower + weak trend (ADX < 25) + price above 1d EMA50
            if (vol_spike[i] and 
                close[i] < bb_lower[i] and 
                adx[i] < 25 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: volume spike + price above BB upper + weak trend (ADX < 25) + price below 1d EMA50
            elif (vol_spike[i] and 
                  close[i] > bb_upper[i] and 
                  adx[i] < 25 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: trend strengthens OR price reverts to VWAP
        elif position == 1:
            # Exit long: ADX > 30 OR price crosses above VWAP
            if (adx[i] > 30 or 
                close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX > 30 OR price crosses below VWAP
            if (adx[i] > 30 or 
                close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Volume_Spike_Reversion_v1"
timeframe = "4h"
leverage = 1.0