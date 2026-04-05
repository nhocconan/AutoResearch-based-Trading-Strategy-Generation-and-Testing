#!/usr/bin/env python3
"""
Experiment #8074: 1-hour timeframe with 4h/1d trend filter and volume confirmation.
Hypothesis: For 1h timeframe, use 4h Donchian breakout for direction and 1d EMA for trend filter,
with volume confirmation on 1h to reduce noise. This should yield 15-37 trades/year by requiring
multiple confirmations, suitable for challenging 1h timeframe where pure strategies fail.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8074_1h_4h_donchian20_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters - tuned for lower frequency
DONCHIAN_PERIOD = 20          # 4h Donchian channels
EMA_PERIOD = 50               # 1d EMA for trend filter
VOLUME_MA_PERIOD = 20         # 1h volume moving average
VOLUME_THRESHOLD = 1.5        # Volume must be 1.5x average
ATR_PERIOD = 14               # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5     # Wider stop for 1h volatility
SIGNAL_SIZE = 0.20            # Conservative position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators (1h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average (1h)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management (1h)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(highest_high[i-1]) or np.isnan(lowest_low[i-1]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Volume confirmation on 1h
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions on 4h channels - require close beyond channel bands
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions - require ALL confirmations
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
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