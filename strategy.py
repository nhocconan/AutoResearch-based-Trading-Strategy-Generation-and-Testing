#!/usr/bin/env python3
"""
exp_6737_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with daily EMA(50) trend filter and volume confirmation.
In trending markets (price > EMA50): long on upper band breakout, short on lower band breakout.
In ranging markets (price near EMA50): fade at Donchian extremes with volume spike.
Daily EMA provides structural trend filter to avoid counter-trend whipsaws in bear markets.
Volume confirmation ensures breakouts/mean reversions are legitimate.
Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6737_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for daily EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
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
                
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on daily EMA
        # Trending market: price significantly above/below EMA
        # Ranging market: price near EMA
        ema_distance_pct = abs(close[i] - ema_1d_aligned[i]) / ema_1d_aligned[i] if ema_1d_aligned[i] != 0 else 0
        is_trending = ema_distance_pct > 0.02  # 2% distance from EMA
        is_ranging = ema_distance_pct <= 0.02   # Within 2% of EMA
        
        # Trending market: Donchian breakout continuation
        long_breakout = is_trending and (close[i] > highest_high[i]) and vol_confirmed and (close[i] > ema_1d_aligned[i])
        short_breakout = is_trending and (close[i] < lowest_low[i]) and vol_confirmed and (close[i] < ema_1d_aligned[i])
        
        # Ranging market: mean reversion at Donchian extremes
        long_mean_revert = is_ranging and (close[i] <= lowest_low[i]) and vol_confirmed
        short_mean_revert = is_ranging and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout or long_mean_revert:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout or short_mean_revert:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals