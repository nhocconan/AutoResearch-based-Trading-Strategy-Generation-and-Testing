#!/usr/bin/env python3
"""
exp_6680_4h_donchian20_1d_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1-day EMA trend filter and volume confirmation.
In ranging markets (price between 1d EMA±1.5*ATR), fade to EMA. In trending markets
(price outside 1d EMA±2.0*ATR), breakout in EMA direction with volume confirmation.
Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year) with discretionary
position sizing (0.25) to minimize fee drag. Works in both bull (breakouts) and bear
(mean reversion to EMA) regimes.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6680_4h_donchian20_1d_ema_vol_v1"
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
MAX_HOLD_BARS = 6  # ~1 day (4h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Calculate 1-day ATR for regime definition
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    # Align HTF indicators to LTF (4h) with shift(1) for completed days only
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(highest[i]) or np.isnan(lowest[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine market regime based on 1d EMA and ATR
        # Trending: price outside EMA ± 2.0*ATR
        # Ranging: price inside EMA ± 1.5*ATR
        ema_upper = ema_1d_aligned[i] + 2.0 * atr_1d_aligned[i]
        ema_lower = ema_1d_aligned[i] - 2.0 * atr_1d_aligned[i]
        ema_upper_weak = ema_1d_aligned[i] + 1.5 * atr_1d_aligned[i]
        ema_lower_weak = ema_1d_aligned[i] - 1.5 * atr_1d_aligned[i]
        
        strong_uptrend = close[i] > ema_upper
        strong_downtrend = close[i] < ema_lower
        ranging_market = (close[i] >= ema_lower_weak) and (close[i] <= ema_upper_weak)
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Mean reversion signals (in ranging market)
        long_mean_revert = ranging_market and (close[i] <= ema_1d_aligned[i])
        short_mean_revert = ranging_market and (close[i] >= ema_1d_aligned[i])
        
        # Breakout signals (in trending market)
        long_breakout = strong_uptrend and (close[i] > highest[i]) and vol_confirmed
        short_breakout = strong_downtrend and (close[i] < lowest[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_mean_revert or long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_mean_revert or short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals