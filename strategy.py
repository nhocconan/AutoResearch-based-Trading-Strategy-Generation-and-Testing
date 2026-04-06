#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12754_1h_triple_barrier_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
ATR_MULT = 2.0
VOLUME_MULT = 2.5
SIGNAL_SIZE = 0.20
SESSION_START = 8
SESSION_END = 20

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    
    # Calculate 1d EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h ATR for stops
    atr_1h = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume filter
    volume_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-calculate session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup
    start = max(ATR_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Session filter: 08-20 UTC
        if not (SESSION_START <= hours[i] <= SESSION_END):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 1d data not ready
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
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
        
        # Volatility regime: only trade when 1d ATR is elevated
        vol_regime = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-100):i+1])
        
        # Trend filter: price above/below 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume spike
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_MULT) if not np.isnan(volume_ma[i]) else False
        
        # Mean reversion signal: fade moves away from EMA when volatile
        if position == 0:
            if vol_regime and volume_spike:
                # Fade extreme moves: short when far above EMA, long when far below
                if above_ema and (close[i] - ema_1d_aligned[i]) > (atr_1d_aligned[i] * 1.5):
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    stop_price = entry_price + (atr_1h[i] * ATR_MULT)
                elif below_ema and (ema_1d_aligned[i] - close[i]) > (atr_1d_aligned[i] * 1.5):
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    stop_price = entry_price - (atr_1h[i] * ATR_MULT)
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals