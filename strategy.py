#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h TRIX zero-line crossover with 1d EMA200 trend filter and volume confirmation
    # TRIX (15,9,9) captures momentum with reduced whipsaw vs MACD
    # Long when TRIX crosses above zero AND price > 1d EMA200 (bullish regime)
    # Short when TRIX crosses below zero AND price < 1d EMA200 (bearish regime)
    # Volume spike (>2.0x 30-period average) confirms institutional participation
    # Session filter (08-20 UTC) reduces low-liquidity noise trades
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 12h close: TRIX = EMA(EMA(EMA(close, 15), 9), 9)
    # First EMA(15)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA(9)
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Third EMA(9) = TRIX
    trix = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12h volume confirmation (>2.0x 30-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_12h[i] = np.mean(volume[i-30:i])
    volume_spike_12h = volume > (2.0 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)  # TRIX is 12h indicator, no alignment needed actually
    # Correction: TRIX is calculated on 12h data, so no HTF alignment needed
    trix_aligned = trix  # Already on 12h timeframe
    
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_aligned[i-1]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike_12h[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # TRIX zero-line crossover
        trix_cross_above = trix_aligned[i-1] <= 0 and trix_aligned[i] > 0
        trix_cross_below = trix_aligned[i-1] >= 0 and trix_aligned[i] < 0
        
        # 1d trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: TRIX crossover + trend alignment + volume confirmation
        long_entry = trix_cross_above and bullish_trend and volume_spike_12h[i]
        short_entry = trix_cross_below and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: opposite TRIX crossover or trend reversal
        long_exit = trix_cross_below or not bullish_trend
        short_exit = trix_cross_above or not bearish_trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_trix_zero_cross_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0