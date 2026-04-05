#!/usr/bin/env python3
"""
exp_7554_1h_ema200_4h_donchian_1d_atrvol
Hypothesis: 1-hour EMA200 trend filter combined with 4-hour Donchian(20) breakout and 1-day volume confirmation.
Trades in direction of higher timeframe trend to avoid whipsaws. Volume confirms breakout strength.
Position size fixed at 0.20 (20% of capital) to limit drawdown. Session filter (08-20 UTC) reduces noise.
Targets 60-150 trades over 4 years (15-37/year) with strict multi-timeframe confluence.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7554_1h_ema200_4h_donchian_1d_atrvol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_TREND = 200
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.20

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_4h_200 = pd.Series(close_4h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    
    # 1d volume MA for confirmation
    volume_1d = df_1d['volume'].values
    volume_1d_ma = pd.Series(volume_1d).ewm(span=VOLUME_MA_PERIOD, adjust=False, min_periods=VOLUME_MA_PERIOD).mean().values
    volume_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_ma)
    
    # 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high = pd.Series(high_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low_4h).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    # LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(ema_4h_200_aligned[i]) or np.isnan(volume_1d_ma_aligned[i]) or \
           np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]):
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
        
        # Determine market regime from 4h EMA200
        bull_regime = close[i] > ema_4h_200_aligned[i]
        bear_regime = close[i] < ema_4h_200_aligned[i]
        
        # Volume confirmation (1-day volume MA)
        volume_confirmed = volume[i] > (volume_1d_ma_aligned[i] * VOLUME_THRESHOLD) if not np.isnan(volume_1d_ma_aligned[i]) else False
        
        # Breakout conditions from 4h Donchian
        upper_breakout = high[i] > highest_high_aligned[i-1] if i-1 >= 0 and not np.isnan(highest_high_aligned[i-1]) else False
        lower_breakout = low[i] < lowest_low_aligned[i-1] if i-1 >= 0 and not np.isnan(lowest_low_aligned[i-1]) else False
        
        # Entry conditions
        long_entry = bull_regime and upper_breakout and volume_confirmed
        short_entry = bear_regime and lower_breakout and volume_confirmed
        
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