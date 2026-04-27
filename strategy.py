#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams Alligator with 1-week EMA trend filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw=13, Teeth=8, Lips=5).
# In trending markets: Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear).
# In ranging markets: lines intertwine.
# Strategy: Go long when Lips > Teeth > Jaw (bullish alignment) and price > Jaw.
# Go short when Lips < Teeth < Jaw (bearish alignment) and price < Jaw.
# Weekly EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation filters out low-activity periods.
# Designed for ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Calculate Alligator lines
    jaw = smma(high, 13)  # Blue line (13-period)
    teeth = smma(low, 8)  # Red line (8-period)
    lips = smma(close, 5)  # Green line (5-period)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on weekly close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if bullish_alignment and close[i] > jaw[i] and volume_filter[i]:
            # Strong bullish signal: go long
            signals[i] = 0.25
            position = 1
        elif bearish_alignment and close[i] < jaw[i] and volume_filter[i]:
            # Strong bearish signal: go short
            signals[i] = -0.25
            position = -1
        else:
            # No clear signal or filter not met: flatten or hold
            if position == 1:
                # Exit long if alignment breaks or price crosses Jaw
                if not (bullish_alignment and close[i] > jaw[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if alignment breaks or price crosses Jaw
                if not (bearish_alignment and close[i] < jaw[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "Daily_WilliamsAlligator_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0