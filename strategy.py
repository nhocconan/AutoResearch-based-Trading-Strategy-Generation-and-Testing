#!/usr/bin/env python3
"""
exp_7495_6d_donchian20_1w_pivot_vol_v1
Hypothesis: 6d Donchian(20) breakout with 1-week pivot direction and volume confirmation.
- Long when price breaks above Donchian(20) high, price > 1w pivot point, and volume > 1.5x 20-period average
- Short when price breaks below Donchian(20) low, price < 1w pivot point, and volume > 1.5x 20-period average
- Uses 1-week pivot points calculated from prior week's high/low/close
- Filters out low-volume breakouts to avoid false signals
- Targets 50-150 total trades over 4 years (12-37/year) with strict breakout conditions
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7495_6d_donchian20_1w_pivot_vol_v1"
timeframe = "6d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
VOLUME_AVG_PERIOD = 20
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate pivot points: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w, r1_1w, s1_1w = calculate_pivot_points(high_1w, low_1w, close_1w)
    # Pivot points are for the week that just completed, so we use them for current week
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_avg = pd.Series(volume).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    
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
    start = max(DONCHIAN_PERIOD, VOLUME_AVG_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]):
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
        
        # Volume condition: current volume > VOLUME_MULTIPLIER * average volume
        volume_condition = volume[i] > VOLUME_MULTIPLIER * volume_avg[i]
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > donchian_high[i] and  # price breaks above Donchian high
            close[i] > pivot_1w_aligned[i] and  # price above weekly pivot
            volume_condition  # volume confirmation
        )
        
        bearish_breakout = (
            close[i] < donchian_low[i] and  # price breaks below Donchian low
            close[i] < pivot_1w_aligned[i] and  # price below weekly pivot
            volume_condition  # volume confirmation
        )
        
        # Exit conditions: return to pivot point
        long_exit = close[i] <= pivot_1w_aligned[i]
        short_exit = close[i] >= pivot_1w_aligned[i]
        
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