#!/usr/bin/env python3
"""
exp_7519_6h_donchian20_12h_volume_v1
Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 12h EMA50 trend filter.
Long: Price breaks above 6h Donchian high(20) + 12h volume > 1.5x avg volume + price > 12h EMA50
Short: Price breaks below 6h Donchian low(20) + 12h volume > 1.5x avg volume + price < 12h EMA50
Uses volume surge to confirm breakouts in both bull and bear markets.
Targets 80-150 total trades over 4 years (20-38/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7519_6h_donchian20_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
EMA_TREND = 50
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
    
    # Calculate 12h average volume for volume filter
    volume_12h = df_12h['volume'].values
    avg_volume_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_TREND, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_50_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]):
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
        
        # Breakout conditions
        upper_break = close[i] > donchian_high[i-1]  # break above previous high
        lower_break = close[i] < donchian_low[i-1]   # break below previous low
        
        # Volume confirmation
        volume_surge = volume[i] > VOLUME_MULTIPLIER * avg_volume_12h_aligned[i]
        
        # Trend filter
        uptrend = close[i] > ema_12h_50_aligned[i]
        downtrend = close[i] < ema_12h_50_aligned[i]
        
        # Entry conditions
        long_entry = upper_break and volume_surge and uptrend
        short_entry = lower_break and volume_surge and downtrend
        
        # Exit conditions (opposite breakout or trend reversal)
        long_exit = lower_break or (close[i] < ema_12h_50_aligned[i] and position == 1)
        short_exit = upper_break or (close[i] > ema_12h_50_aligned[i] and position == -1)
        
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