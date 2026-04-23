#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA Trend Filter and Volume Spike
- Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via SMAs with future shift
- 1d EMA50 > EMA200 ensures alignment with strong daily trend for multi-timeframe confirmation
- Volume > 1.5x 20-period average confirms breakout momentum with moderate filtering
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via Alligator alignment with trend, in bear markets via fade when Alligator diverges
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Trend filter: 1 = uptrend (EMA50 > EMA200), -1 = downtrend (EMA50 < EMA200), 0 = no trend
    trend_1d = np.where(ema50_1d > ema200_1d, 1, np.where(ema50_1d < ema200_1d, -1, 0))
    
    # Align trend to 12h timeframe (completed 1d bar only)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (13-period SMMA, shifted 8 bars forward)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # future shift
    
    # Teeth (8-period SMMA, shifted 5 bars forward)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # future shift
    
    # Lips (5-period SMMA, shifted 3 bars forward)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # future shift
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # EMA200 needs 200 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_aligned[i]) or 
            np.isnan(jaw.iloc[i]) or 
            np.isnan(teeth.iloc[i]) or 
            np.isnan(lips.iloc[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        
        # Alligator signals with trend filter and volume confirmation
        # Long: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume spike
        # Short: Lips < Teeth < Jaw (bearish alignment) + downtrend + volume spike
        long_signal = (lips_val > teeth_val > jaw_val and 
                      trend_aligned[i] == 1 and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (lips_val < teeth_val < jaw_val and 
                       trend_aligned[i] == -1 and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator convergence (|Lips-Jaw| < 0.5% of price) or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator convergence or trend turns down
                if (abs(lips_val - jaw_val) < 0.005 * close[i] or 
                    trend_aligned[i] == -1):
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator convergence or trend turns up
                if (abs(lips_val - jaw_val) < 0.005 * close[i] or 
                    trend_aligned[i] == 1):
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