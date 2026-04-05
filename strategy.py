#!/usr/bin/env python3
"""
Experiment #8894: 1h strategy with 4h/1d trend filter + volume confirmation.
Hypothesis: Use 4h/1d timeframes to establish directional bias (trend), 
1h for precise entry timing with volume confirmation, and session filter (08-20 UTC) 
to reduce noise. Target 60-150 total trades over 4 years (15-37/year) to minimize 
fee drag while capturing meaningful moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_8894_1h_4h1d_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (EWMA)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Use pandas EWMA with alpha = 1/period for Wilder's smoothing
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 4h for trend, 1d for higher timeframe bias
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Calculate 1d EMA for higher timeframe bias
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Price relative to EMAs: above = bullish bias, below = bearish bias
    # Require both 4h and 1d to agree for stronger signal
    bullish_4h = close_4h > ema_4h
    bearish_4h = close_4h < ema_4h
    bullish_1d = close_1d > ema_1d
    bearish_1d = close_1d < ema_1d
    
    # Combined bias: both timeframes must agree
    bull_bias = bullish_4h & bullish_1d
    bear_bias = bearish_4h & bearish_1d
    
    # Align HTF bias to 1h timeframe
    bull_bias_aligned = align_htf_to_ltf(prices, df_4h, bull_bias.astype(float))
    bear_bias_aligned = align_htf_to_ltf(prices, df_4h, bear_bias.astype(float))
    
    # Calculate LTF indicators (1h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (reduce noise outside active hours)
    hours = prices.index.hour  # index is already DatetimeIndex
    session_active = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available or outside session
        if np.isnan(bull_bias_aligned[i]) or np.isnan(bear_bias_aligned[i]) or not session_active[i]:
            # Hold current position or flatten if outside session
            if not session_active[i] and position != 0:
                signals[i] = 0.0
                position = 0
            else:
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
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions - require trend alignment + volume
        long_entry = bull_bias_aligned[i] == 1.0 and volume_confirmed
        short_entry = bear_bias_aligned[i] == 1.0 and volume_confirmed
        
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