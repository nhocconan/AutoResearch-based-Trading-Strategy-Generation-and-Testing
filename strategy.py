#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h/1d trend alignment and volume confirmation.
- Uses 4h EMA50 and 1d EMA200 for trend direction (bullish when price > both).
- 1h Donchian(20) breakout for entry timing.
- Volume > 1.5x 20-period average for confirmation.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Fixed size 0.20 to control risk and fees.
- Target: 15-37 trades/year (60-150 over 4 years) by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6454_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    signals = np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # === HTF INDICATORS (computed ONCE before loop) ===
    # 4h EMA50 for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA200 for long-term trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === LTF INDICATORS (pre-computed) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h Donchian Channel (20)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume average (20)
    vol_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    
    # === SIGNAL GENERATION LOOP ===
    position = 0  # 0: flat, 1: long
    entry_price = 0.0
    
    for i in range(200, n):  # Start after warmup
        # Skip if not in session
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0
            if position == 1:
                position = 0
            continue
        
        # Skip if indicators not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or \
           np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0
            if position == 1:
                position = 0
            continue
        
        # === ENTRY CONDITIONS (LONG ONLY) ===
        # 1. Trend alignment: price > 4h EMA50 AND price > 1d EMA200
        trend_up = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        
        # 2. Donchian breakout: price breaks above 20-period high
        donchian_break = close[i] > highest[i]
        
        # 3. Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # 4. Price above Donchian mid-range (avoid false breakouts in weak trends)
        donchian_mid = (highest[i] + lowest[i]) / 2
        price_above_mid = close[i] > donchian_mid
        
        if position == 0:
            # Enter long if all conditions met
            if trend_up and donchian_break and vol_confirm and price_above_mid:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            else:
                signals[i] = 0
        else:  # position == 1
            # Exit conditions
            # Stoploss: 2.5 * ATR(14) approximation using 20-period range
            atr_approx = (highest[i] - lowest[i]) / 2  # Rough proxy
            stop_price = entry_price - 2.5 * atr_approx
            
            # Trend reversal: price below 4h EMA50 OR 1d EMA200
            trend_down = close[i] < ema_4h_aligned[i] or close[i] < ema_1d_aligned[i]
            
            # Donchian breakdown: price breaks below 20-period low
            donchian_breakdown = close[i] < lowest[i]
            
            if stop_price > 0 and (close[i] < stop_price or trend_down or donchian_breakdown):
                signals[i] = 0.0  # Exit
                position = 0
            else:
                signals[i] = 0.20  # Hold
    
    return signals