#!/usr/bin/env python3
"""
exp_6711_6h_adx_di_crossover_1d_v1
Hypothesis: 6h ADX + DI crossover with 1-day trend filter. Uses ADX(14) > 25 to confirm trend strength,
DI+ > DI- for long signals, DI- > DI+ for short signals, aligned with 1-day EMA50 direction.
Only trades in direction of higher timeframe trend to avoid whipsaws. Designed for 6h timeframe
to capture multi-day trends with minimal fee drag (~20-40 trades/year expected).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6711_6h_adx_di_crossover_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
DI_PERIOD = 14
ADX_THRESHOLD = 25.0
EMA_PERIOD_1D = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD_1D, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX and DI calculation
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    # Directional Movement
    up_move = pd.Series(high - np.roll(high, 1))
    down_move = pd.Series(np.roll(low, 1) - low)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_smoothed = pd.Series(tr).ewm(span=ADX_PERIOD, adjust=False).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).ewm(span=ADX_PERIOD, adjust=False).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=ADX_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(ADX_PERIOD * 2, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if indicators not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
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
            
        # Determine trend direction from 1d EMA
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # DI crossover signals
        di_cross_up = plus_di[i] > minus_di[i]
        di_cross_down = minus_di[i] > plus_di[i]
        
        # Previous DI for crossover detection
        prev_plus_di = plus_di[i-1] if i > 0 else plus_di[i]
        prev_minus_di = minus_di[i-1] if i > 0 else minus_di[i]
        di_cross_up_prev = prev_plus_di <= prev_minus_di
        di_cross_down_prev = prev_minus_di <= prev_plus_di
        
        # Long signal: DI+ crosses above DI- in uptrend with strong ADX
        long_signal = (di_cross_up and di_cross_up_prev and 
                      uptrend_1d and strong_trend)
        
        # Short signal: DI- crosses above DI+ in downtrend with strong ADX
        short_signal = (di_cross_down and di_cross_down_prev and
                       downtrend_1d and strong_trend)
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
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