#!/usr/bin/env python3
"""
exp_7138_1d_donchian20_1w_hma_v1
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter.
In trending markets (price > HMA21): Donchian breakouts in trend direction.
In ranging markets (price near HMA21): mean reversion at Donchian bands with volume confirmation.
Uses 1w HMA for regime and 1d Donchian for entries/exits.
Designed for 1d timeframe to capture swings with ~7-25 trades/year (30-100 total over 4 years).
Works in both bull and bear markets by adapting to HMA-defined trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7138_1d_donchian20_1w_hma_v1"
timeframe = "1d"
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
    
    # Load HTF data ONCE before loop - using 1w for HMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA (Hull Moving Average)
    close_1w = df_1w['close'].values
    half_period = HMA_PERIOD // 2
    sqrt_period = int(np.sqrt(HMA_PERIOD))
    
    # WMA function
    def wma(values, period):
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    wma_half = np.full_like(close_1w, np.nan)
    wma_full = np.full_like(close_1w, np.nan)
    
    for i in range(half_period - 1, len(close_1w)):
        wma_half[i] = wma(close_1w[i - half_period + 1:i + 1], half_period)
    
    for i in range(HMA_PERIOD - 1, len(close_1w)):
        wma_full[i] = wma(close_1w[i - HMA_PERIOD + 1:i + 1], HMA_PERIOD)
    
    raw_hma = 2 * wma_half - wma_full
    hma_1w = np.full_like(close_1w, np.nan)
    
    for i in range(sqrt_period - 1, len(raw_hma)):
        if not np.isnan(raw_hma[i - sqrt_period + 1:i + 1]).any():
            hma_1w[i] = wma(raw_hma[i - sqrt_period + 1:i + 1], sqrt_period)
    
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
                
        # Determine market regime based on HMA
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Mean reversion conditions (touch bands and reverse)
        touch_upper = abs(close[i] - highest_high[i]) < (highest_high[i] - lowest_low[i]) * 0.02
        touch_lower = abs(close[i] - lowest_low[i]) < (highest_high[i] - lowest_low[i]) * 0.02
        
        # Entry logic
        if position == 0:
            # In bull regime: trade breakouts up, fade touches down
            if bull_regime:
                if breakout_up and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif touch_lower and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
            # In bear regime: trade breakouts down, fade touches up
            elif bear_regime:
                if breakout_down and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif touch_upper and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
            # In transition regime (near HMA): only trade with strong volume confirmation
            else:
                if breakout_up and vol_confirmed:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_down and vol_confirmed:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals