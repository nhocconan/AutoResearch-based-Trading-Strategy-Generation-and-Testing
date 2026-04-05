#!/usr/bin/env python3
"""
exp_7140_4h_donchian20_1d_hma_v1
Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter + volume confirmation.
In trending markets (price above/below 1d HMA21): take Donchian breakouts in trend direction.
In ranging markets (price near 1d HMA21): avoid false breakouts, reducing whipsaw.
Uses 1d HMA for regime filter and 4h volume for confirmation. Designed for 4h timeframe
to capture swings with ~19-50 trades/year (75-200 total over 4 years).
HMA reduces lag vs EMA/SMA, improving trend detection in both bull and bear markets.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7140_4h_donchian20_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
HMA_PERIOD = 21
MAX_HOLD_BARS = 6  # ~6 * 4h = 1 day

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for HMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA (Hull Moving Average)
    close_1d = df_1d['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1d, half_period)
    wma_full = wma(close_1d, HMA_PERIOD)
    raw_hma = 2 * wma_half - wma_full
    hma_1d = wma(raw_hma, sqrt_period)
    
    # Align to LTF (4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
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
        if np.isnan(hma_1d_aligned[i]):
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
        
        # Determine market regime based on 1d HMA
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high[i]
        breakout_short = close[i] < lowest_low[i]
        
        # Enter long: price above HMA + bullish breakout + volume
        if price_above_hma and breakout_long and vol_confirmed:
            if position == 0:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = position * SIGNAL_SIZE
        # Enter short: price below HMA + bearish breakout + volume
        elif price_below_hma and breakout_short and vol_confirmed:
            if position == 0:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = position * SIGNAL_SIZE
        # Exit conditions: price crosses HMA in opposite direction
        elif position == 1 and price_below_hma:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and price_above_hma:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        # Hold current position
        else:
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
    
    return signals