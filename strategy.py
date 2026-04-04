#!/usr/bin/env python3
"""
exp_6713_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
In ranging markets (price between Donchian bands), fade extremes toward 12h HMA mean.
In trending markets (break Donchian bands), continue in breakout direction with volume confirmation.
12h HMA trend filter avoids counter-trend trades. Designed for 4h timeframe to capture
swings while minimizing fee drag (~19-50 trades/year expected).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6713_4h_donchian20_12h_hma_v1"
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
MAX_HOLD_BARS = 6  # ~1 day (4h bars)

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = pd.Series(series).rolling(window=half_period, min_periods=half_period).apply(
        lambda x: np.dot(x, weights_half) / weights_half.sum(), raw=True
    ).values
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = pd.Series(series).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights_full) / weights_full.sum(), raw=True
    ).values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
        lambda x: np.dot(x, weights_sqrt) / weights_sqrt.sum(), raw=True
    ).values
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, HMA_PERIOD)
    
    # Align HTF HMA to LTF (4h) with shift(1) for completed bars only
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
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
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
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine market regime based on 12h HMA
        # Above HMA = uptrend bias, Below HMA = downtrend bias
        uptrend_bias = close[i] > hma_12h_aligned[i]
        downtrend_bias = close[i] < hma_12h_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Mean reversion signals (fade Donchian extremes toward HMA)
        long_mean_revert = (close[i] <= lowest_low[i]) and (close[i] < hma_12h_aligned[i])
        short_mean_revert = (close[i] >= highest_high[i]) and (close[i] > hma_12h_aligned[i])
        
        # Breakout signals (continue in direction of HMA trend)
        long_breakout = (close[i] >= highest_high[i]) and uptrend_bias and vol_confirmed
        short_breakout = (close[i] <= lowest_low[i]) and downtrend_bias and vol_confirmed
        
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