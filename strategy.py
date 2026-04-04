#!/usr/bin/env python3
"""
exp_6687_6h_donchian20_1d_ema_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1-day EMA(50) trend filter and volume confirmation.
In uptrend (price > 1d EMA50), long on breakout above 20-period high with volume surge.
In downtrend (price < 1d EMA50), short on breakdown below 20-period low with volume surge.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 6h timeframe
to capture multi-day swings while avoiding overtrading (~15-30 trades/year expected).
Works in both bull (breakouts continue) and bear (breakdowns continue) markets via trend filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6687_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_SURGE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_ma = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_ma = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for surge confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for dynamic stoploss
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
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if indicators not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema_aligned[i]
        downtrend = close[i] < ema_aligned[i]
        
        # Volume confirmation (surge above average)
        vol_surge = volume[i] > vol_ma[i] * VOL_SURGE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout signals
        breakout_long = uptrend and (close[i] > high_ma[i]) and vol_surge
        breakdown_short = downtrend and (close[i] < low_ma[i]) and vol_surge
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakdown_short:
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