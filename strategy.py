#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
- Long: Jaw < Teeth < Lips (bullish alignment) + price > 1d EMA50 + volume > 1.8x 20-period average
- Short: Jaw > Teeth > Lips (bearish alignment) + price < 1d EMA50 + volume > 1.8x 20-period average
- Exit: Alligator lines cross (Teeth crosses Jaw) or trend reverses
- Uses Williams Alligator (SMAs with specific offsets) for trend identification
- 1d EMA50 filter ensures alignment with higher timeframe trend
- Volume spike confirms momentum strength
- Discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Williams Alligator works in both bull and bear markets by showing clear trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: Jaw (13-period SMA, offset 8), Teeth (8-period SMA, offset 5), Lips (5-period SMA, offset 3)
    # Using median price (typical price) as input
    typical_price = (high + low + close) / 3.0
    
    # Jaw: 13-period SMA of typical price, shifted by 8 bars
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift right by 8 (offset into future)
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMA of typical price, shifted by 5 bars
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift right by 5 (offset into future)
    teeth[:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips: 5-period SMA of typical price, shifted by 3 bars
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift right by 3 (offset into future)
    lips[:3] = np.nan  # first 3 values invalid due to shift
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # EMA50 needs 50, volume MA needs 20, Jaw needs 13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Williams Alligator signals with trend filter and volume confirmation
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Long: Bullish alignment + uptrend + volume spike
        long_signal = bullish_alignment and uptrend and (volume[i] > 1.8 * vol_ma[i])
        # Short: Bearish alignment + downtrend + volume spike
        short_signal = bearish_alignment and downtrend and (volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment or trend reversal
                if bearish_alignment or not uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bullish alignment or trend reversal
                if bullish_alignment or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0