#!/usr/bin/env python3
"""
exp_6689_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1-day EMA filter and volume confirmation.
Long when price breaks above 20-period Donchian high + price > 1d EMA50 + volume > 1.5x MA.
Short when price breaks below 20-period Donchian low + price < 1d EMA50 + volume > 1.5x MA.
Uses ATR(14) stoploss at 2.0. Designed for 4h timeframe to capture medium-term trends
with minimal overtrading (target: 75-200 trades over 4 years).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6689_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD)
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
                
        # Breakout conditions with filters
        bullish_breakout = (close[i] > donchian_high[i-1]) and (close[i] > ema_1d_aligned[i])
        bearish_breakout = (close[i] < donchian_low[i-1]) and (close[i] < ema_1d_aligned[i])
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        # Enter new positions only if flat
        if position == 0:
            if bullish_breakout and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_breakout and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals