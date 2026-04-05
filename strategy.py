#!/usr/bin/env python3
"""
Experiment #8794: 1h momentum with 4h/1d trend filter + volume + session filter.
Hypothesis: Combining 4h trend direction with 1d trend filter reduces false signals,
while volume confirmation and session filter (08-20 UTC) avoid low-liquidity noise.
Using 1h for entry timing with momentum (price > open + 0.5*ATR) ensures clean entries.
Target: 60-150 trades over 4 years (15-37/year) to balance opportunity and fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8794_1h_momentum_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
ATR_PERIOD = 14
MOMENTUM_THRESHOLD = 0.5  # price must exceed open by 0.5*ATR
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h close > open (bullish candle)
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    bull_4h = close_4h > open_4h  # True if bullish candle
    
    # Calculate 1d close > open (bullish candle)
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    bull_1d = close_1d > open_1d  # True if bullish candle
    
    # Align HTF signals to LTF
    bull_4h_aligned = align_htf_to_ltf(prices, df_4h, bull_4h.astype(float))
    bull_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_1d.astype(float))
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # ATR for momentum and stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ATR_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(bull_4h_aligned[i]) or np.isnan(bull_1d_aligned[i]):
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
        
        # Determine trend bias from 4h and 1d (both must agree)
        bullish_bias = bull_4h_aligned[i] > 0.5 and bull_1d_aligned[i] > 0.5
        bearish_bias = bull_4h_aligned[i] < 0.5 and bull_1d_aligned[i] < 0.5
        
        # Momentum condition: price > open + threshold*ATR (long) or < open - threshold*ATR (short)
        price_vs_open = close[i] - open_price[i]
        long_momentum = price_vs_open > (MOMENTUM_THRESHOLD * atr[i])
        short_momentum = price_vs_open < (-MOMENTUM_THRESHOLD * atr[i])
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Session filter: 08-20 UTC
        in_session = 8 <= hours[i] <= 20
        
        # Entry conditions
        long_entry = bullish_bias and long_momentum and volume_confirmed and in_session
        short_entry = bearish_bias and short_momentum and volume_confirmed and in_session
        
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
</x>