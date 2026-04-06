#!/usr/bin/env python3
"""
4h ADX + Donchian Breakout with Volume Confirmation
Hypothesis: In trending markets (ADX > 25), price breaks Donchian channels with volume follow through.
Long on upper band breakout, short on lower band breakout. Uses 1d trend filter to avoid counter-trend trades.
Works in bull (catch breakouts) and bear (short breakdowns). Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14346_4h_adx_donchian_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for daily trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ADX (14) for trend strength
    adx_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr_ma = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_di = 100 * plus_dm_ma / (tr_ma + 1e-10)
    minus_di = 100 * minus_dm_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.8 * vol_ma)  # Require at least 80% of average volume
    
    # ATR for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, adx_period*2) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below lower Donchian band OR stoploss OR trend weakens
            if close[i] <= lower[i] or close[i] <= entry_price - 2.5 * atr[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above upper Donchian band OR stoploss OR trend weakens
            if close[i] >= upper[i] or close[i] >= entry_price + 2.5 * atr[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + strong trend + volume + trend filter
            long_breakout = close[i] > upper[i]
            short_breakout = close[i] < lower[i]
            strong_trend = adx[i] > 25
            trend_up = close[i] > ema_1d_aligned[i]
            trend_down = close[i] < ema_1d_aligned[i]
            
            if long_breakout and strong_trend and trend_up and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and strong_trend and trend_down and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals