#!/usr/bin/env python3
"""
exp_7113_4h_donchian20_12h_hma_v1
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
In ranging markets (price between Donchian bands): mean revert at band touches with volume spike.
In trending markets (breaks Donchian(20) with 12h HMA alignment): continuation in breakout direction.
Uses 12h HMA for trend direction and 4h volume for confirmation.
Designed for 4h timeframe to capture swings with ~19-50 trades/year (75-200 total over 4 years).
Works in both bull and bear markets by adapting to 12h HMA trend regime.
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
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 6  # ~6 * 4h = 1 day
HMA_PERIOD = 21

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for HMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA (Hull Moving Average)
    close_12h = df_12h['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        if period <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.full_like(close_12h, np.nan)
    wma_full = np.full_like(close_12h, np.nan)
    
    for i in range(half_period, len(close_12h)):
        wma_half[i] = wma(close_12h[i-half_period+1:i+1], half_period)
    for i in range(HMA_PERIOD, len(close_12h)):
        wma_full[i] = wma(close_12h[i-HMA_PERIOD+1:i+1], HMA_PERIOD)
    
    hma_12h = np.full_like(close_12h, np.nan)
    for i in range(HMA_PERIOD, len(close_12h)):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma = 2 * wma_half[i] - wma_full[i]
            if i >= half_period + sqrt_period - 1:
                hma_12h[i] = wma(np.full(sqrt_period, raw_hma), sqrt_period) if not np.isnan(raw_hma) else np.nan
    
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
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on 12h HMA
        hma_trend_up = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_trend_down = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        # Mean reversion at Donchian bands in ranging/weak trend
        mean_revert_long = (close[i] <= lowest_low[i]) and vol_confirmed and not hma_trend_down
        mean_revert_short = (close[i] >= highest_high[i]) and vol_confirmed and not hma_trend_up
        
        # Continuation breakouts with trend alignment
        continuation_long = (close[i] > highest_high[i]) and vol_confirmed and hma_trend_up
        continuation_short = (close[i] < lowest_low[i]) and vol_confirmed and hma_trend_down
        
        # Enter new positions only if flat
        if position == 0:
            if mean_revert_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif mean_revert_short or continuation_short:
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

</think>