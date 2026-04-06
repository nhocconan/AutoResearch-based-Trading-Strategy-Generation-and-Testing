#!/usr/bin/env python3
"""
1h mean-reversion with 4h trend filter and volume confirmation
Hypothesis: In 4h trends, 1h pullbacks to VWAP or mean offer high-probability entries.
Use 4h EMA50 for trend direction, 1h RSI(14) for oversold/overbought, and volume spike for confirmation.
Trades only during active London/New York session (08-20 UTC) to avoid low-volume noise.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_meanrev_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_prev = np.roll(ema50_4h, 1)
    ema50_4h_prev[0] = ema50_4h[0]
    ema50_rising = ema50_4h > ema50_4h_prev
    ema50_falling = ema50_4h < ema50_4h_prev
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) - Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detector: current volume > 2.0 * 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 60  # For RSI and EMA
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI mean reversion or stoploss
        if position == 1:  # long position
            # Exit: RSI > 60 (overbought) OR stoploss
            if (rsi[i] > 60 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 40 (oversold) OR stoploss
            if (rsi[i] < 40 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + trend + volume spike
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            
            long_entry = rsi_oversold and ema50_rising_aligned[i] and vol_spike[i]
            short_entry = rsi_overbought and ema50_falling_aligned[i] and vol_spike[i]
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals