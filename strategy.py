#!/usr/bin/env python3
"""
Experiment #9480: 4h Donchian20 + HMA Trend + Volume Spike + ATR Stoploss.
Hypothesis: Donchian channel breakouts (20-period) on 4h timeframe, 
confirmed by HMA trend direction and volume spikes, provide high-probability 
trend-following entries. Works in bull markets via breakouts and bear markets 
via breakdowns. Targets 75-200 total trades over 4 years (19-50/year) to 
balance opportunity and cost. Uses 1h trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9480_4h_donchian20_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_PERIOD = 21
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    n = int(period)
    if n < 1:
        return close
    half_n = n // 2
    sqrt_n = int(np.sqrt(n))
    
    # WMA function
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    # Calculate WMAs
    wma_half = wma(close, half_n)
    wma_full = wma(close, n)
    
    # Calculate raw HMA
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    # Pad to original length
    result = np.full_like(close, np.nan, dtype=float)
    start_idx = n - half_n
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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1h for trend filter)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 1h HMA for trend filter
    hma_1h = calculate_hma(df_1h['close'].values, HMA_PERIOD)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 4h Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HMA data not available
        if np.isnan(hma_1h_aligned[i]):
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
        
        # Trend filter: HMA direction
        hma_rising = hma_1h_aligned[i] > hma_1h_aligned[i-1] if i > 0 else False
        hma_falling = hma_1h_aligned[i] < hma_1h_aligned[i-1] if i > 0 else False
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Entry conditions
        long_entry = breakout_up and hma_rising and volume_spike
        short_entry = breakout_down and hma_falling and volume_spike
        
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