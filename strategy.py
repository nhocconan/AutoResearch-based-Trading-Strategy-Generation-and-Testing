#!/usr/bin/env python3
"""
exp_7314_1h_donchian20_4h_ema_vol_v1
Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) trend filter and volume confirmation.
Uses 4h EMA to define trend regime: above EMA = continuation breakouts, near EMA = mean reversion at extremes.
Volume confirmation filters false breakouts. Session filter (08-20 UTC) reduces noise.
Designed for 1h timeframe targeting 60-150 trades over 4 years (15-37/year).
Works in bull markets via continuation breakouts and bear markets via mean reversion at Donchian extremes.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7314_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~1 day

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 4h for EMA trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Align to LTF (1h)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]):
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
            
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on EMA
        above_ema = close[i] > ema_4h_aligned[i]
        below_ema = close[i] < ema_4h_aligned[i]
        near_ema = np.abs(close[i] - ema_4h_aligned[i]) < (0.5 * atr[i])  # Within 0.5 ATR of EMA
        
        # Fade at extremes in ranging market (near EMA)
        fade_long = near_ema and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = near_ema and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market
        continuation_long = above_ema and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_ema and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat and in session
        if position == 0 and in_session:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position or flat outside session
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals