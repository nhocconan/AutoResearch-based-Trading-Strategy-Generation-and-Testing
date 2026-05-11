#!/usr/bin/env python3
"""
4H_Donchian_20_Breakout_TRIX_Zero_Cross_Volume_Confirmation
Hypothesis: 4h Donchian(20) breakouts capture trend continuation. TRIX(12) zero cross confirms momentum shift.
Volume spike (>1.5x 20-period EMA) filters false breakouts. Works in bull/bear by taking both long and short breakouts.
Target: 25-40 trades/year (~100-160 total over 4 years) to stay under 400 trade limit and minimize fee drag.
"""

name = "4H_Donchian_20_Breakout_TRIX_Zero_Cross_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === TRIX CALCULATION (12-period) ===
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3  # percentage change
    trix = np.where(np.isnan(trix) | (ema3 == 0), 0, trix)  # handle div by zero
    
    # === VOLUME FILTER ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === DONCHIAN CHANNEL (20-period) ===
    # Highest high and lowest low over past 20 periods
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    # For true rolling window, we need to look back 20 periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # For first 20 bars, use expanding window
    for i in range(20):
        donchian_high[i] = highest_high[i]
        donchian_low[i] = lowest_low[i]
    
    # === TRIX ZERO CROSS SIGNAL ===
    # TRIX > 0 = bullish momentum, TRIX < 0 = bearish momentum
    trix_bullish = trix > 0
    trix_bearish = trix < 0
    
    # === SIGNAL GENERATION ===
    position_size = 0.25
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian + some for TRIX stability)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trix[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high[i]
        breakout_short = close[i] < donchian_low[i]
        
        if position == 0:
            # Long: Donchian breakout up + TRIX bullish + volume spike
            if breakout_long and trix_bullish[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Donchian breakout down + TRIX bearish + volume spike
            elif breakout_short and trix_bearish[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - use opposite Donchian band or TRIX reversal
            if position == 1:
                # Exit long: price breaks below Donchian low OR TRIX turns bearish
                if close[i] < donchian_low[i] or (trix[i] < 0 and trix[i-1] >= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: price breaks above Donchian high OR TRIX turns bullish
                if close[i] > donchian_high[i] or (trix[i] > 0 and trix[i-1] <= 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals