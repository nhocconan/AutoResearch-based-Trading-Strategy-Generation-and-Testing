#!/usr/bin/env python3
"""
exp_7113_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
Long when price breaks above Donchian upper band with 12h HMA rising and volume > 1.5x MA.
Short when price breaks below Donchian lower band with 12h HMA falling and volume > 1.5x MA.
ATR(14) stoploss at 2.5x. Designed for 4h timeframe to capture swings with ~19-50 trades/year.
Works in both bull and bear markets by using HMA trend filter to avoid counter-trend breakouts.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7113_4h_donchian20_12h_hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
HMA_PERIOD = 21

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA
    close_12h = df_12h['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        if period <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    wma_half = wma(close_12h, half_period)
    wma_full = wma(close_12h, HMA_PERIOD)
    hma_12h = 2 * wma_half - wma_full
    hma_12h = wma(hma_12h, sqrt_period)
    
    # Align to LTF (4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
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
        if np.isnan(hma_12h_aligned[i]):
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
                
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # HMA trend direction (rising/falling)
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Donchian breakouts
        breakout_up = close[i] > highest_high[i-1] if i > 0 else False
        breakout_down = close[i] < lowest_low[i-1] if i > 0 else False
        
        # Enter new positions only if flat
        if position == 0:
            if breakout_up and hma_rising and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_down and hma_falling and vol_confirmed:
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