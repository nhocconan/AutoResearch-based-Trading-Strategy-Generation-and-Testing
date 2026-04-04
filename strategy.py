#!/usr/bin/env python3
"""
exp_6454_1h_donchian20_4h_1d_ema_vol_v1
- Hypothesis: 1h Donchian(20) breakout with 4h EMA200 trend filter and 1d EMA50 regime filter.
  Volume confirmation (>1.5x avg volume) reduces false breakouts.
  Session filter (08-20 UTC) avoids low-liquidity periods.
  Uses 4h/1d for signal direction, 1h only for entry timing.
  Discrete position sizing (0.20) to minimize fee churn.
  Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6454_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours once
    hours = prices.index.hour
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA200 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h_200 = close_4h.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    
    # 1d EMA50 for regime filter (bull/bear)
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d_50 = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # 1h indicators (computed once, vectorized)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 1h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = max(100, 200)  # ensure EMA200 is valid
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0  # exit outside session
                position = 0
            continue
        
        # Skip if indicators not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema_4h_200_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or \
           np.isnan(vol_ma[i]):
            continue
        
        # Regime: 1d EMA50 - bull if price > EMA50, bear if price < EMA50
        bull_regime = close[i] > ema_1d_50_aligned[i]
        bear_regime = close[i] < ema_1d_50_aligned[i]
        
        # 4h trend filter
        uptrend_4h = close[i] > ema_4h_200_aligned[i]
        downtrend_4h = close[i] < ema_4h_200_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry logic
        if position == 0:
            # Long: bull regime + 4h uptrend + breakout up + volume
            if bull_regime and uptrend_4h and breakout_up and vol_ok:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: bear regime + 4h downtrend + breakout down + volume
            elif bear_regime and downtrend_4h and breakout_down and vol_ok:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
        
        # Exit logic
        elif position == 1:  # Long position
            # Stoploss: 2*ATR approximation using 20-period range
            atr_approx = (highest_high[i] - lowest_low[i]) / 2.0
            if close[i] < entry_price - 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            # Take profit: exit at 3*ATR profit
            elif close[i] > entry_price + 3.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            # Reverse signal
            elif bear_regime and downtrend_4h and breakout_down and vol_ok:
                signals[i] = -0.20  # reverse to short
                position = -1
                entry_price = close[i]
            # Exit outside session
            elif hour < 8 or hour > 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            atr_approx = (highest_high[i] - lowest_low[i]) / 2.0
            if close[i] > entry_price + 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            elif close[i] < entry_price - 3.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            elif bull_regime and uptrend_4h and breakout_up and vol_ok:
                signals[i] = 0.20  # reverse to long
                position = 1
                entry_price = close[i]
            elif hour < 8 or hour > 20:
                signals[i] = 0.0
                position = 0
    
    return signals