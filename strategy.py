#!/usr/bin/env python3
"""
exp_7414_1h_volume_breakout_4h_trend_v1
Hypothesis: 1h volume breakout with 4h trend filter (EMA) to capture momentum in both bull and bear markets.
Uses 4h EMA for direction, 1h volume breakout for entry timing to limit trades (target: 60-150/4 years).
Volume filter ensures only strong moves trigger entries. Works in bull (breakouts up) and bear (breakdowns down).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7414_1h_volume_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 20
EMA_SLOW = 50
VOL_MA_PERIOD = 20
VOL_BREAKOUT_MULT = 2.0
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5
MAX_HOLD_BARS = 24  # 24 hours max hold

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 4h EMA for trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMAs
    close_4h = df_4h['close'].values
    ema_fast_4h = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow_4h = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # Align to LTF (1h)
    ema_fast_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_fast_4h)
    ema_slow_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_slow_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for breakout detection
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(EMA_SLOW, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_fast_4h_aligned[i]) or np.isnan(ema_slow_4h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULT * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume breakout condition
        vol_breakout = volume[i] > vol_ma[i] * VOL_BREAKOUT_MULT if not np.isnan(vol_ma[i]) else False
        
        # Determine 4h trend
        uptrend = ema_fast_4h_aligned[i] > ema_slow_4h_aligned[i]
        downtrend = ema_fast_4h_aligned[i] < ema_slow_4h_aligned[i]
        
        # Enter long on uptrend + volume breakout up
        # Enter short on downtrend + volume breakout down
        if position == 0:
            if uptrend and vol_breakout and close[i] > open[i]:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif downtrend and vol_breakout and close[i] < open[i]:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals