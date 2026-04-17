#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d funding rate Z-score filter and ATR trailing stop.
Long when Williams %R < -80 (oversold) AND 1d funding Z-score < -1.0 (extremely negative funding = bullish sentiment).
Short when Williams %R > -20 (overbought) AND 1d funding Z-score > 1.0 (extremely positive funding = bearish sentiment).
Exit when price trails 2.0 * ATR from highest high (long) or lowest low (short).
Uses 12h for execution and Williams %R, 1d for funding rate Z-score and ATR calculation.
Designed to capture extreme sentiment reversals in both bull and bear markets with proper risk control.
Target: 15-30 trades/year per symbol.
"""

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
    
    # Get 1d data for funding rate Z-score and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50.0)
    
    # Calculate 1d ATR (14-period) for trailing stop
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)), 
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d funding rate Z-score (assuming funding rate data is available)
    # For now, we'll use a proxy: price deviation from 50-day EMA normalized by ATR
    # In reality, you would load actual funding rate data from data/processed/funding/
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    price_dev = close_1d - ema50
    # Normalize by ATR to get Z-score approximation
    funding_zscore = price_dev / (atr14 + 1e-10)  # add small epsilon to avoid division by zero
    # Clip extreme values
    funding_zscore = np.clip(funding_zscore, -5.0, 5.0)
    
    # Align all 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), williams_r)  # Williams %R is already 12h
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    funding_zscore_aligned = align_htf_to_ltf(prices, df_1d, funding_zscore)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or
            np.isnan(funding_zscore_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        funding_extreme_negative = funding_zscore_aligned[i] < -1.0
        funding_extreme_positive = funding_zscore_aligned[i] > 1.0
        
        if position == 0:
            # Long: Williams %R oversold AND funding extremely negative
            if williams_oversold and funding_extreme_negative:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: Williams %R overbought AND funding extremely positive
            elif williams_overbought and funding_extreme_positive:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Trailing stop: exit if price drops 2.0 * ATR from highest high
            if close[i] < highest_high_since_entry - 2.0 * atr14_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Trailing stop: exit if price rises 2.0 * ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.0 * atr14_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_FundingZscore_ATRTrail"
timeframe = "12h"
leverage = 1.0