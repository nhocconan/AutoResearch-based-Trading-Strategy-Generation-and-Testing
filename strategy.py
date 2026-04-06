#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions, 1d EMA provides trend filter,
# volume spike confirms momentum. Works in bull/bear by fading extremes in trend direction.
# Target: 80-160 trades over 4 years (20-40/year) to balance opportunity and cost.

name = "exp_12991_6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
WILLIAMS_OVERBOUGHT = -20
WILLIAMS_OVERSOLD = -80
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_williams_r(high, low, close, period):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r.fillna(0).values

def calculate_ema(values, period):
    """EMA calculation"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
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
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMAs for trend filter
    close_d = df_daily['close'].values
    ema_fast_d = calculate_ema(close_d, EMA_FAST_PERIOD)
    ema_slow_d = calculate_ema(close_d, EMA_SLOW_PERIOD)
    
    # Trend: 1 = uptrend (fast > slow), -1 = downtrend (fast < slow)
    trend_d = np.where(ema_fast_d > ema_slow_d, 1, -1)
    
    # Align trend to 6h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_daily, trend_d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, EMA_SLOW_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if trend not available
        if np.isnan(trend_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Williams %R signals with trend filter
        williams_oversold = williams_r[i] <= WILLIAMS_OVERSOLD
        williams_overbought = williams_r[i] >= WILLIAMS_OVERBOUGHT
        
        # Long: oversold in uptrend
        long_signal = williams_oversold and (trend_aligned[i] == 1) and volume_ok
        # Short: overbought in downtrend
        short_signal = williams_overbought and (trend_aligned[i] == -1) and volume_ok
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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