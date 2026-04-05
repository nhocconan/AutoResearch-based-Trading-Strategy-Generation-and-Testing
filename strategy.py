#!/usr/bin/env python3
"""
exp_7500_4h_donchian20_1d_vol_ema_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 filter and volume confirmation.
In bull markets (price > 1d EMA50): buy breakouts above 20-period high.
In bear markets (price < 1d EMA50): sell breakdowns below 20-period low.
Uses volume > 1.5x 20-period average to filter false breakouts.
Targets 80-180 trades over 4 years (20-45/year) with strict breakout conditions + volume filter.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7500_4h_donchian20_1d_vol_ema_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 50
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    # Using pandas rolling for clarity and proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = low_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).mean().values
    
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
        if np.isnan(ema_1d_50_aligned[i]):
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
        above_ema50 = close[i] > ema_1d_50_aligned[i]  # bull regime
        below_ema50 = close[i] < ema_1d_50_aligned[i]  # bear regime
        
        # Volume confirmation
        vol_confirm = volume[i] > (VOLUME_MULTIPLIER * volume_ma[i]) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > donchian_high[i] and  # price breaks above 20-period high
            above_ema50 and                  # bull regime
            vol_confirm                      # volume confirmation
        )
        
        bearish_breakout = (
            close[i] < donchian_low[i] and   # price breaks below 20-period low
            below_ema50 and                  # bear regime
            vol_confirm                      # volume confirmation
        )
        
        # Exit conditions (reverse signal)
        long_exit = close[i] < donchian_low[i]  # price breaks below 20-period low
        short_exit = close[i] > donchian_high[i]  # price breaks above 20-period high
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_breakout:
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