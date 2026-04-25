#!/usr/bin/env python3
"""
4h Camarilla H4L4 Breakout with 1d HMA34 Trend and Volume Spike
Hypothesis: Camarilla H4/L4 levels represent stronger intraday support/resistance than H3/L3.
Breakouts above H4 or below L4 with volume spike and aligned 1d HMA34 trend capture significant
institutional moves with lower frequency and higher reliability. Designed for 20-40 trades/year
on 4h to work in both bull (trend following) and bear (mean reversion via exits) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
    # WMA of full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA34 trend and Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period HMA on 1d close for trend
    hma_34_1d = calculate_hma(df_1d['close'].values, 34)
    hma_34_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_34_1d)
    
    # Calculate Camarilla pivots for each 1d bar: based on previous day's high, low, close
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Use previous day's data to calculate today's levels (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla formulas:
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to LTF (4h) - no extra delay needed as pivots are based on completed 1d bar
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for HMA, volume MA, and to avoid NaN from shift
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(hma_34_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        hma_trend = hma_34_1d_aligned[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H4 resistance AND volume spike AND price > 1d HMA34 (uptrend)
            long_entry = (curr_close > h4_level) and vol_spike and (curr_close > hma_trend)
            # Short: price breaks below L4 support AND volume spike AND price < 1d HMA34 (downtrend)
            short_entry = (curr_close < l4_level) and vol_spike and (curr_close < hma_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below L4 support (broken support) OR price crosses below HMA (trend change)
            if (curr_close < l4_level) or (curr_close < hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above H4 resistance (broken resistance) OR price crosses above HMA (trend change)
            if (curr_close > h4_level) or (curr_close > hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dHMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0