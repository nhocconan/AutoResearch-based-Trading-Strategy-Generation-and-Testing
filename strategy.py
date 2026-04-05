#!/usr/bin/env python3
"""
exp_7138_1d_donchian20_1w_hma_v1
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter.
Long when price breaks above Donchian(20) high AND 1w HMA rising.
Short when price breaks below Donchian(20) low AND 1w HMA falling.
Volume confirmation reduces false breakouts.
ATR-based stoploss and 5-day max hold control risk.
Designed for 1d timeframe to capture major swings with ~7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by following 1w HMA trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7138_1d_donchian20_1w_hma_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 5  # 5 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA (Hull Moving Average)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    # Align to LTF (1d)
    hma_aligned = align_htf_to_ltf(prices, df_1w, hma)
    
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(hma_aligned[i]):
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Check for Donchian breakouts
        bull_breakout = close[i] > highest_high[i]
        bear_breakout = close[i] < lowest_low[i]
        
        # Check HMA trend (rising/falling)
        hma_rising = hma_aligned[i] > hma_aligned[i-1]
        hma_falling = hma_aligned[i] < hma_aligned[i-1]
        
        # Enter new positions only if flat
        if position == 0:
            # Long: bull breakout + rising HMA + volume
            if bull_breakout and hma_rising and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: bear breakout + falling HMA + volume
            elif bear_breakout and hma_falling and vol_confirmed:
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