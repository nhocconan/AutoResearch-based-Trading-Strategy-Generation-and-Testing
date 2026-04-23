#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA Trend Filter and Volume Spike
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via smoothed SMAs
- 1d EMA50 > EMA200 ensures alignment with strong daily trend for multi-timeframe confirmation
- Volume > 1.5x 20-period average confirms breakout momentum with strict filtering
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via trend continuation, in bear markets via mean reversion at extremes
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 12h timeframe (completed 1d bar only)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # EMA200 needs 200 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals with trend filter and volume confirmation
        # Alligator is sleeping (all lines intertwined) -> no trade
        # Alligator is awake with mouth open (lines separated) -> trade in direction of alignment
        # Long: Lips > Teeth > Jaw (bullish alignment) + EMA50 > EMA200 (uptrend) + volume spike
        # Short: Lips < Teeth < Jaw (bearish alignment) + EMA50 > EMA200 (uptrend filter) + volume spike -> actually we want EMA50 < EMA200 for short
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Long signal: bullish alignment + uptrend + volume spike
        long_signal = bullish_alignment and uptrend and volume_spike
        # Short signal: bearish alignment + downtrend + volume spike
        short_signal = bearish_alignment and downtrend and volume_spike
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator starts to sleep (lines converge) or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator sleeping or trend turns down
                if not bullish_alignment or not uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator sleeping or trend turns up
                if not bearish_alignment or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0