#!/usr/bin/env python3
"""
Experiment #8514: 1h timeframe with 4h/1d trend filter and volume confirmation
Hypothesis: Using 4h trend for direction and 1d for regime filter reduces false signals in choppy markets.
1h provides timely entries while higher timeframes filter noise. Target 15-37 trades/year to avoid fee drag.
Volume confirmation ensures institutional participation. Works in bull (trend follows) and bear (avoids counter-trend).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8514_1h_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA trend (fast/slow crossover)
    close_4h = df_4h['close'].values
    ema_4h_fast = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_4h_slow = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    ema_4h_crossover = np.where(ema_4h_fast > ema_4h_slow, 1, -1)  # 1=bullish, -1=bearish
    ema_4h_crossover_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_crossover)
    
    # Calculate 1d RSI for regime filter (avoid overextended conditions)
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, RSI_PERIOD)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, RSI_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_4h_crossover_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 4h EMA crossover
        bull_bias = ema_4h_crossover_aligned[i] == 1   # 4h fast above slow
        bear_bias = ema_4h_crossover_aligned[i] == -1  # 4h fast below slow
        
        # 1d RSI regime filter (avoid overbought/oversold extremes)
        rsi_not_overbought = rsi_1d_aligned[i] < RSI_OVERBOUGHT
        rsi_not_oversold = rsi_1d_aligned[i] > RSI_OVERSOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and rsi_not_overbought and volume_confirmed
        short_entry = bear_bias and rsi_not_oversold and volume_confirmed
        
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