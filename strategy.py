#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily trend filter and volume confirmation
# Uses daily timeframe for trend direction to avoid whipsaws in choppy markets.
# Entry only when price breaks Donchian channel with volume above average.
# Stop loss via ATR-based trailing stop (implemented as signal=0 when price < highest - 2*ATR).
# Designed for low frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

name = "12h_donchian20_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for stop loss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    highest_since_long = 0  # Track highest high since entering long
    lowest_since_short = 0  # Track lowest low since entering short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        # Update trailing stop levels
        if position == 1:
            highest_since_long = max(highest_since_long, high[i])
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low[i])
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if trend turns down OR price drops 2*ATR from peak
            if not daily_uptrend or close[i] < highest_since_long - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if trend turns up OR price rises 2*ATR from trough
            if not daily_downtrend or close[i] > lowest_since_short + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: daily uptrend + price breaks above Donchian high + volume confirmation
            if daily_uptrend and close[i] > highest_high[i] and vol_confirm:
                position = 1
                highest_since_long = high[i]
                signals[i] = 0.25
            # Enter short: daily downtrend + price breaks below Donchian low + volume confirmation
            elif daily_downtrend and close[i] < lowest_low[i] and vol_confirm:
                position = -1
                lowest_since_short = low[i]
                signals[i] = -0.25
    
    return signals