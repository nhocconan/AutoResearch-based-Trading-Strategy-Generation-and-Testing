#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA200 Trend Filter and Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA200 for trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Williams %R(14) for extreme oversold/overbought conditions: long when %R crosses above -80 from below,
  short when %R crosses below -20 from above.
- Volume confirmation: current volume > 2.0 * volume MA(20) to avoid low-liquidity false signals.
- Exit: Close-based reversal - exit long when %R crosses below -50 from above,
        exit short when %R crosses above -50 from below.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Designed for both bull and bear markets: Williams %R captures mean reversion in extremes,
  EMA200 filter ensures alignment with higher timeframe trend, volume spike confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    df_1d_close = df_1d['close'].values
    ema_200 = pd.Series(df_1d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF EMA200 to 6h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 200, 14, 20)  # Need enough bars for EMA200, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Williams %R crosses above -80 from below AND 1d EMA200 bullish AND volume confirmed
            if (curr_williams_r > -80 and prev_williams_r <= -80 and 
                curr_close > ema_200_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND 1d EMA200 bearish AND volume confirmed
            elif (curr_williams_r < -20 and prev_williams_r >= -20 and 
                  curr_close < ema_200_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses below -50 from above
            if curr_williams_r < -50 and prev_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses above -50 from below
            if curr_williams_r > -50 and prev_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0