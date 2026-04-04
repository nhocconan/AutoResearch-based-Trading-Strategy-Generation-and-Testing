#!/usr/bin/env python3
"""
exp_6693_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian channel breakout with 12h HMA trend filter and volume confirmation.
Only take breakouts in direction of 12h HMA(21) trend. Uses ATR-based stoploss.
Designed for 4h timeframe to capture medium-term swings while minimizing fee drag (~20-50 trades/year expected).
Works in both bull and bear markets by only trading with the higher timeframe trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6693_4h_donchian20_12h_hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
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
    
    # Calculate 12h HMA(21) - Hull Moving Average
    close_12h = df_12h['close'].values
    half_len = int(21 / 2)
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    wma_2x_sub = 2 * wma_half[-len(wma_full):] - wma_full
    if len(wma_2x_sub) >= sqrt_len:
        hma_12h = wma(wma_2x_sub, sqrt_len)
    else:
        hma_12h = np.full_like(close_12h, np.nan)
    
    # Align HTF HMA to LTF (4h) with shift(1) for completed bars only
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(hma_12h_aligned[i])):
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
                
        # Determine trend direction from 12h HMA
        # For first value, assume no trend
        if i == start:
            hma_trend = 0  # neutral
        else:
            hma_trend = 1 if hma_12h_aligned[i] > hma_12h_aligned[i-1] else -1
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD
        
        # Breakout conditions
        long_breakout = (close[i] > donchian_high[i-1]) and vol_confirmed and (hma_trend == 1)
        short_breakout = (close[i] < donchian_low[i-1]) and vol_confirmed and (hma_trend == -1)
        
        # Exit conditions: reverse signal or stoploss (handled above)
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