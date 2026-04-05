#!/usr/bin/env python3
"""
exp_7513_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 4h upper Donchian + 12h EMA50 up + volume > 1.5x average.
Short when price breaks below 4h lower Donchian + 12h EMA50 down + volume > 1.5x average.
Uses ATR-based stoploss and position sizing of 0.25. Targets 75-200 trades over 4 years.
Designed to work in both bull and bear markets by following the 12h trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7513_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 50
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h_50 = pd.Series(close_12h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    high_roll = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    low_roll = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_50_aligned[i]):
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
        
        # Determine trend direction
        uptrend = ema_12h_50_aligned[i] > ema_12h_50_aligned[i-1] if i > 0 else False
        downtrend = ema_12h_50_aligned[i] < ema_12h_50_aligned[i-1] if i > 0 else False
        
        # Volume confirmation
        vol_confirm = volume[i] > VOLUME_MULTIPLIER * avg_volume[i] if not np.isnan(avg_volume[i]) else False
        
        # Entry conditions
        long_entry = (
            close[i] > high_roll[i-1] and  # break above upper Donchian
            uptrend and                    # 12h EMA50 rising
            vol_confirm                    # volume confirmation
        )
        
        short_entry = (
            close[i] < low_roll[i-1] and   # break below lower Donchian
            downtrend and                  # 12h EMA50 falling
            vol_confirm                    # volume confirmation
        )
        
        # Exit conditions
        long_exit = close[i] < low_roll[i-1]  # break below lower Donchian
        short_exit = close[i] > high_roll[i-1]  # break above upper Donchian
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals