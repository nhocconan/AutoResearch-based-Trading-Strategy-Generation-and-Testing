#!/usr/bin/env python3
"""
Experiment #8374: 1-hour momentum with 4h/1d trend filter and volume confirmation.
Hypothesis: In ranging/bear markets (2025+), price often reverts to the mean after sharp moves. 
Using 4h trend (EMA50) and 1d trend (EMA200) as filters, we enter on 1h pullbacks to the 4h EMA20 
with volume confirmation. This reduces whipsaw and captures mean reversion moves. 
Session filter (08-20 UTC) avoids low-liquidity hours. Target: 60-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8374_1h_meanrev_4h1d_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_4H_PERIOD = 50
EMA_1D_PERIOD = 200
EMA_1H_FAST = 20
EMA_1H_SLOW = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SESSION_START = 8   # UTC
SESSION_END = 20    # UTC

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=EMA_4H_PERIOD, adjust=False, min_periods=EMA_4H_PERIOD).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_1D_PERIOD, adjust=False, min_periods=EMA_1D_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMAs for entry
    ema_fast = pd.Series(close).ewm(span=EMA_1H_FAST, adjust=False, min_periods=EMA_1H_FAST).mean().values
    ema_slow = pd.Series(close).ewm(span=EMA_1H_SLOW, adjust=False, min_periods=EMA_1H_SLOW).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Pre-compute session hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_4H_PERIOD, EMA_1D_PERIOD, EMA_1H_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < SESSION_START or hour > SESSION_END:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if HTF data not available
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]):
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
        
        # Determine trend bias from 4h and 1d EMA
        # Bullish: price above both EMAs
        # Bearish: price below both EMAs
        # Neutral: mixed signals (avoid trading)
        price_above_4h = close[i] > ema_4h_aligned[i]
        price_above_1d = close[i] > ema_1d_aligned[i]
        bullish_bias = price_above_4h and price_above_1d
        bearish_bias = (not price_above_4h) and (not price_above_1d)
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # 1h EMA crossover for entry timing
        fast_above_slow = ema_fast[i] > ema_slow[i]
        fast_below_slow = ema_fast[i] < ema_slow[i]
        
        # Entry conditions: pullback to 4h EMA with volume and alignment
        long_entry = bullish_bias and fast_above_slow and volume_confirmed and (close[i] > ema_4h_aligned[i] * 0.995)
        short_entry = bearish_bias and fast_below_slow and volume_confirmed and (close[i] < ema_4h_aligned[i] * 1.005)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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