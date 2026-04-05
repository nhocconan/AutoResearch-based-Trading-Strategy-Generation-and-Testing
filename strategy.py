#!/usr/bin/env python3
"""
exp_7512_12h_1d_1w_donchian_vol
Hypothesis: 12h Donchian breakout with daily volume confirmation and weekly trend filter.
In bull markets (price > weekly EMA50): long on upper band breakout with volume spike.
In bear markets (price < weekly EMA50): short on lower band breakout with volume spike.
Uses daily volume > 1.5x 20-period average as confirmation. Targets 50-150 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7512_12h_1d_1w_donchian_vol"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
EMA_WEEKLY = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=VOLUME_PERIOD, min_periods=VOLUME_PERIOD).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_WEEKLY, adjust=False, min_periods=EMA_WEEKLY).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_PERIOD, EMA_WEEKLY, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(vol_avg_1d_aligned[i]) or np.isnan(ema_1w_50_aligned[i]):
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
        
        # Determine market regime
        bull_regime = close[i] > ema_1w_50_aligned[i]   # price above weekly EMA50
        bear_regime = close[i] < ema_1w_50_aligned[i]   # price below weekly EMA50
        
        # Volume confirmation
        volume_spike = volume[i] > VOLUME_MULTIPLIER * vol_avg_1d_aligned[i]
        
        # Entry conditions
        long_entry = (
            bull_regime and           # bull market
            close[i] > highest_high[i] and  # break above upper Donchian
            volume_spike              # volume confirmation
        )
        
        short_entry = (
            bear_regime and           # bear market
            close[i] < lowest_low[i] and  # break below lower Donchian
            volume_spike              # volume confirmation
        )
        
        # Exit conditions - reverse signal on opposite breakout
        long_exit = close[i] < lowest_low[i]  # exit long on lower band break
        short_exit = close[i] > highest_high[i]  # exit short on upper band break
        
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