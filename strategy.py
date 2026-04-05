# Investigate 6h: donchian(20) + 1d POC + volume
# Pivot point: use 1d high/low/close to compute POC as (H+L+C)/3
# Long when price > POC and breaks Donchian high + volume
# Short when price < POC and breaks Donchian low + volume
# Trend filter: require price > 1d EMA200 for longs, < for shorts
# Risk: ATR stop 2x, target 3x
# Position size 0.25
# Expect ~150-250 trades over 4 years

#!/usr/bin/env python3
from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8071_6h_donchian20_1d_poc_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_LONG = 200  # for trend filter on 1d
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d POC: (H+L+C)/3
    poc_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=EMA_LONG, adjust=False, min_periods=EMA_LONG).mean().values
    
    # POC relative to EMA: if POC > EMA200, bullish bias; else bearish
    # Actually we'll use price vs POC for entry, EMA for filter
    ema_filter = np.where(close_1d > ema_200, 1, -1)  # 1=bullish bias (price above EMA200), -1=bearish
    ema_filter_aligned = align_htf_to_ltf(prices, df_1d, ema_filter)
    poc_aligned = align_htf_to_ltf(prices, df_1d, poc_1d)
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, EMA_LONG) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(poc_aligned[i]) or np.isnan(ema_filter_aligned[i]):
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
        
        # Determine bias from 1d EMA200
        bull_bias = ema_filter_aligned[i] == 1   # 1d close above EMA200
        bear_bias = ema_filter_aligned[i] == -1  # 1d close below EMA200
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond channel bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_bias and (close[i] > poc_aligned[i]) and upper_breakout and volume_confirmed
        short_entry = bear_bias and (close[i] < poc_aligned[i]) and lower_breakout and volume_confirmed
        
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