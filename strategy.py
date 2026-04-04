#!/usr/bin/env python3
"""
exp_6703_4h_donchian20_12h_hma_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
In bull markets: buy breakouts above 20-period high when 12h HMA is rising and volume > 1.5x MA.
In bear markets: sell breakdowns below 20-period low when 12h HMA is falling and volume > 1.5x MA.
Uses ATR-based stoploss (2x ATR) to manage risk. Designed for 4h timeframe to capture
medium-term trends while minimizing fee drag (~20-50 trades/year expected).
Works in both bull and bear markets by following the 12h HMA trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6703_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA (Hull Moving Average)
    close_12h = df_12h['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA calculation
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_period)
    wma_full = wma(close_12h, HMA_PERIOD)
    raw_hma = 2 * wma_half - wma_full
    hma_12h = wma(raw_hma, sqrt_period)
    
    # Align HTF HMA to LTF (4h) with shift(1) for completed bars only
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    def rolling_max(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = np.max(arr[i - period + 1:i + 1])
        return result
    
    def rolling_min(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period - 1, len(arr)):
            result[i] = np.min(arr[i - period + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, DONCHIAN_PERIOD)
    donchian_low = rolling_min(low, DONCHIAN_PERIOD)
    
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
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
                
        # Determine trend direction from 12h HMA (rising/falling)
        # Use previous bar to avoid look-ahead
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Breakout signals
        long_breakout = close[i] > donchian_high[i] and hma_rising and vol_confirmed
        short_breakout = close[i] < donchian_low[i] and hma_falling and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals