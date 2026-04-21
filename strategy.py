#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_ChopRegime_ATRStop_V1
Hypothesis: 4h KAMA (adaptive trend) with volume confirmation (>1.5x 20-period volume MA) and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Uses 1d HTF EMA50 for trend bias (price > EMA50 longs, < EMA50 shorts). ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR. Designed for low trade frequency (<200 total 4h trades) to minimize fee drag and work in both bull/bear markets via regime adaptation. KAMA adapts to market noise, reducing false signals in choppy conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # KAMA (adaptive moving average) - 30-period, fast=2, slow=30
    close_series = pd.Series(close_4h)
    direction = np.abs(close_series - close_series.shift(10))
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        if np.isnan(sc[i]) or sc[i] == 0:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_sum = tr.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(chop_sum / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) 
            or np.isnan(chop[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Regime detection
        is_choppy = chop[i] > 61.8  # mean reversion regime
        is_trending = chop[i] < 38.2  # trend following regime
        
        if position == 0:
            # Long: Price > KAMA + volume + trend filter (in uptrend or choppy market)
            if price > kama[i] and vol_ok and (price > ema_50_1d_aligned[i] or is_choppy):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price < KAMA + volume + trend filter (in downtrend or choppy market)
            elif price < kama[i] and vol_ok and (price < ema_50_1d_aligned[i] or is_choppy):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Price < KAMA or loss of volume/momentum
            elif price < kama[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Price > KAMA or loss of volume/momentum
            elif price > kama[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Volume_ChopRegime_ATRStop_V1"
timeframe = "4h"
leverage = 1.0