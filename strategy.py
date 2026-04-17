#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Channel_v1
Breakout strategy using Keltner Channels (EMA20 + 2*ATR10) with volume confirmation and ADX trend filter.
Enters long when price closes above upper KC with ADX>25 and volume>1.5x average.
Enters short when price closes below lower KC with ADX>25 and volume>1.5x average.
Exits when price crosses back through EMA20 or ADX falls below 20.
Designed to capture trends with volatility-adjusted breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === EMA20 for Keltner Channel center ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === True Range and ATR(10) for Keltner Channel width ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Keltner Channel bounds ===
    kc_upper = ema20 + 2 * atr10
    kc_lower = ema20 - 2 * atr10
    
    # === ADX(14) for trend strength ===
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(ema20[i]) or 
            np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or 
            np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price closes above upper KC, ADX > 25, volume confirmed, price above 1d EMA50
            if (close[i] > kc_upper[i] and 
                adx[i] > 25 and 
                vol_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price closes below lower KC, ADX > 25, volume confirmed, price below 1d EMA50
            elif (close[i] < kc_lower[i] and 
                  adx[i] > 25 and 
                  vol_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: trend weakening or reversal through EMA20
        elif position == 1:
            # Exit long: ADX < 20 OR price closes below EMA20
            if (adx[i] < 20 or 
                close[i] < ema20[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ADX < 20 OR price closes above EMA20
            if (adx[i] < 20 or 
                close[i] > ema20[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_Breakout_Channel_v1"
timeframe = "4h"
leverage = 1.0