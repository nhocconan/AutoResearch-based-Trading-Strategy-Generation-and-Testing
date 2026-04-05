#!/usr/bin/env python3
"""
exp_7574_1h_trend_following_v1
Hypothesis: Use 4h trend (EMA50 vs EMA200) for directional bias, 1h for entry timing.
Enter long when 4h EMA50 > EMA200 (bullish) and price crosses above 1h EMA20.
Enter short when 4h EMA50 < EMA200 (bearish) and price crosses below 1h EMA20.
Add 8-20 UTC session filter to avoid low-liquidity hours.
ATR-based stop loss (2x) to manage risk.
Target: 60-150 trades over 4 years (15-37/year) with strict trend alignment.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7574_1h_trend_following_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 50
EMA_SLOW = 200
EMA_ENTRY = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.20

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    ema_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_fast)
    ema_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slow)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h EMA20 for entry timing
    ema_20 = pd.Series(close).ewm(span=EMA_ENTRY, adjust=False, min_periods=EMA_ENTRY).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, EMA_ENTRY, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_fast_aligned[i]) or np.isnan(ema_4h_slow_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine 4h trend
        bull_trend = ema_4h_fast_aligned[i] > ema_4h_slow_aligned[i]
        bear_trend = ema_4h_fast_aligned[i] < ema_4h_slow_aligned[i]
        
        # Entry conditions: price crossing EMA20
        price_above_ema20 = close[i] > ema_20[i]
        price_below_ema20 = close[i] < ema_20[i]
        
        # Generate signals
        if position == 0:
            if bull_trend and price_above_ema20:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bear_trend and price_below_ema20:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals