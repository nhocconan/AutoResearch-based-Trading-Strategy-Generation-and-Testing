#!/usr/bin/env python3
"""
exp_7130_1d_donchian20_1w_hma_v1
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter for BTC/ETH/SOL.
In trending markets (price above/below weekly HMA): breakout continuation.
In ranging markets: mean reversion at Donchian extremes with volume confirmation.
Uses weekly HMA for regime filter and daily Donchian for entries/exits.
Designed for 1d timeframe to capture swings with ~7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by adapting to weekly trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7130_1d_donchian20_1w_hma_v1"
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
MAX_HOLD_BARS = 10  # ~10 days

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for HMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA (Hull Moving Average)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA(period) = 2 * WMA(half_period) - WMA(full_period)
    def wma(values, period):
        if period <= 0:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    wma_half = wma(close_1w, half_period)
    wma_full = wma(close_1w, HMA_PERIOD)
    raw_hma = 2 * wma_half - wma_full
    hma_1w = wma(raw_hma, sqrt_period)
    
    # Align to LTF (1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
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
        if np.isnan(hma_1w_aligned[i]):
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
        
        # Determine market regime based on weekly HMA
        above_weekly_hma = close[i] > hma_1w_aligned[i]
        below_weekly_hma = close[i] < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high[i]
        breakout_short = close[i] < lowest_low[i]
        
        # Mean reversion at Donchian extremes
        mean_revert_long = close[i] <= lowest_low[i] and vol_confirmed
        mean_revert_short = close[i] >= highest_high[i] and vol_confirmed
        
        # Entry logic based on regime
        if position == 0:
            # In uptrend (above weekly HMA): look for long breakouts or mean reversion at lows
            if above_weekly_hma:
                if breakout_long or mean_revert_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
            # In downtrend (below weekly HMA): look for short breakouts or mean reversion at highs
            elif below_weekly_hma:
                if breakout_short or mean_revert_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            # In transition (exactly at HMA): use mean reversion only
            else:
                if mean_revert_long:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif mean_revert_short:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals