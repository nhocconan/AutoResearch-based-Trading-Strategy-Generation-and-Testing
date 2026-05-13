#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50.
# Exit on ATR(14) trailing stop (2.0x). Uses 4h primary timeframe and 1d HTF for trend alignment.
# Donchian channels provide robust price structure, EMA50 filters intermediate trend,
# and ATR-based sizing adapts to volatility. Designed for BTC/ETH with moderate entry frequency.

name = "4h_Donchian20_1dEMA50_ATR_Sizing_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate ATR(14) for position sizing and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA50 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 4h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) channels from previous 20 periods
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align to current bar (breakout vs previous channel)
    donchian_upper = np.roll(high_ma, 1)  # Previous 20-period high
    donchian_lower = np.roll(low_ma, 1)   # Previous 20-period low
    # Handle first value
    donchian_upper[0] = np.nan
    donchian_lower[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian upper AND price > 1d EMA50
            if close[i] > donchian_upper[i] and close[i] > ema50_1d_aligned[i]:
                # ATR-based sizing: 0.30 when ATR low, 0.15 when ATR high
                atr_ratio = atr[i] / close[i]  # Normalized ATR
                if atr_ratio < 0.02:  # Low volatility
                    size = 0.30
                elif atr_ratio > 0.05:  # High volatility
                    size = 0.15
                else:  # Medium volatility
                    size = 0.22
                signals[i] = size
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price < Donchian lower AND price < 1d EMA50
            elif close[i] < donchian_lower[i] and close[i] < ema50_1d_aligned[i]:
                # ATR-based sizing: 0.30 when ATR low, 0.15 when ATR high
                atr_ratio = atr[i] / close[i]  # Normalized ATR
                if atr_ratio < 0.02:  # Low volatility
                    size = 0.30
                elif atr_ratio > 0.05:  # High volatility
                    size = 0.15
                else:  # Medium volatility
                    size = 0.22
                signals[i] = -size
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                # Maintain position
                atr_ratio = atr[i] / close[i]  # Normalized ATR
                if atr_ratio < 0.02:  # Low volatility
                    size = 0.30
                elif atr_ratio > 0.05:  # High volatility
                    size = 0.15
                else:  # Medium volatility
                    size = 0.22
                signals[i] = size
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                # Maintain position
                atr_ratio = atr[i] / close[i]  # Normalized ATR
                if atr_ratio < 0.02:  # Low volatility
                    size = 0.30
                elif atr_ratio > 0.05:  # High volatility
                    size = 0.15
                else:  # Medium volatility
                    size = 0.22
                signals[i] = -size
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals