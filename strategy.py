#!/usr/bin/env python3
"""
Experiment #8150: Daily Donchian breakout with weekly trend filter and volume confirmation.
Hypothesis: Price breaking beyond 20-day high/low on daily with volume >1.5x 20-day volume MA 
and aligned weekly trend (weekly EMA slope) captures sustained moves with appropriate frequency 
for daily timeframe. Weekly trend filter reduces whipsaw in choppy markets while targeting 
30-100 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8150_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
WEEKLY_EMA_PERIOD = 21
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
ATR_TARGET_MULTIPLIER = 3.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA and its slope for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=WEEKLY_EMA_PERIOD, adjust=False, min_periods=WEEKLY_EMA_PERIOD).mean().values
    ema_slope = np.diff(ema_1w, prepend=ema_1w[0])  # positive = upward slope
    ema_slope_signal = np.where(ema_slope > 0, 1, -1)  # 1=uptrend, -1=downtrend
    ema_slope_signal_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_signal)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price channel (Donchian)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, WEEKLY_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_slope_signal_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from weekly EMA slope
        bull_bias = ema_slope_signal_aligned[i] == 1   # weekly EMA trending up
        bear_bias = ema_slope_signal_aligned[i] == -1  # weekly EMA trending down
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond channel bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</s>