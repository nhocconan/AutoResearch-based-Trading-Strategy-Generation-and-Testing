#!/usr/bin/env python3
"""
exp_6709_4h_donchian20_1d_hma_vol_v1
Hypothesis: 4h Donchian(20) breakout with 1-day HMA(21) trend filter and volume confirmation.
Goes long when price breaks above 20-bar high AND 1d HMA is rising AND volume > 1.5x MA.
Goes short when price breaks below 20-bar low AND 1d HMA is falling AND volume > 1.5x MA.
Uses ATR(14) stoploss at 2.0x. Designed for 4h timeframe to capture trends while minimizing fee drag (~20-40 trades/year expected).
Works in both bull (breakouts with volume) and bear (short breakdowns with volume) markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6709_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
HMA_PERIOD = 21

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day HMA (Hull Moving Average)
    close_1d = df_1d['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        if period <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.full_like(close_1d, np.nan)
    wma_full = np.full_like(close_1d, np.nan)
    
    for i in range(half_period - 1, len(close_1d)):
        wma_half[i] = wma(close_1d[i - half_period + 1:i + 1], half_period)
    
    for i in range(HMA_PERIOD - 1, len(close_1d)):
        wma_full[i] = wma(close_1d[i - HMA_PERIOD + 1:i + 1], HMA_PERIOD)
    
    raw_hma = 2 * wma_half - wma_full
    hma_1d = np.full_like(close_1d, np.nan)
    
    for i in range(sqrt_period - 1, len(raw_hma)):
        if not np.isnan(raw_hma[i]):
            hma_1d[i] = wma(raw_hma[i - sqrt_period + 1:i + 1], sqrt_period)
    
    # Align HTF HMA to LTF (4h) with shift(1) for completed days only
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    # HMA slope: rising if current > previous, falling if current < previous
    hma_slope = np.diff(hma_1d_aligned, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
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
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD, HMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if indicators not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(hma_rising[i]) or np.isnan(hma_falling[i])):
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
                
        # Breakout conditions with volume confirmation and trend filter
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        long_breakout = (close[i] > donchian_high[i]) and hma_rising[i] and vol_confirmed
        short_breakout = (close[i] < donchian_low[i]) and hma_falling[i] and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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