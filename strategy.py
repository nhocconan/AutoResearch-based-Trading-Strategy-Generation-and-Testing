# 100% Complete Solution for Experiment #9542
#!/usr/bin/env python3
"""
Experiment #9542: 12h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
Hypothesis: Donchian(20) breakouts combined with HMA(21) trend filter and volume confirmation
provide robust trend-following signals. Works in both bull and bear markets by capturing
breakouts in the direction of the higher timeframe trend. Uses 1d HTF for trend context.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9542_12h_donchian_breakout_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
VOLUME_MA_PERIOD = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.2
SIGNAL_SIZE = 0.25

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(close, weights_half, mode='valid') / weights_half.sum()
    
    # WMA for full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(close, weights_full, mode='valid') / weights_full.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    
    # Pad to original length
    result = np.full_like(close, np.nan)
    start_idx = period - 1
    end_idx = start_idx + len(hma)
    if end_idx <= len(close):
        result[start_idx:end_idx] = hma
    return result

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend context)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d = calculate_hma(df_1d['close'].values, HMA_PERIOD)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, HMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price relative to 1d HMA
        uptrend = close[i] > hma_1d_aligned[i]
        downtrend = close[i] < hma_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i-1] if i > 0 and not np.isnan(donchian_high[i-1]) else False
        breakout_short = close[i] < donchian_low[i-1] if i > 0 and not np.isnan(donchian_low[i-1]) else False
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = breakout_long and uptrend and volume_spike
        short_entry = breakout_short and downtrend and volume_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals