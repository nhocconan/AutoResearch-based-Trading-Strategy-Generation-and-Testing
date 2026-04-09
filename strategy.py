#!/usr/bin/env python3
# 12h_williams_alligator_regime_volume_v1
# Hypothesis: 12h strategy using Williams Alligator (Jaw/Teeth/Lips) with 1d trend filter and volume confirmation.
# Long: Lips > Teeth > Jaw (bullish alignment), close > 1d EMA50, volume > 1.5x 20-period average.
# Short: Lips < Teeth < Jaw (bearish alignment), close < 1d EMA50, volume > 1.5x 20-period average.
# Exit: Opposite Alligator alignment or volume divergence.
# Uses Williams Alligator to identify trend phases and avoid choppy markets.
# Volume confirmation filters weak breakouts. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_williams_alligator_regime_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (12h)
    close_s = pd.Series(close)
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = close_s.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = close_s.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA shifted 3 bars
    lips = close_s.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 1:  # Long position
            # Exit: Bearish alignment OR volume divergence (price up but volume down)
            if bearish_alignment or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bullish alignment OR volume divergence (price down but volume down)
            if bullish_alignment or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Bullish alignment, price above 1d EMA50, volume confirmed
            if (bullish_alignment and close[i] > ema50_1d_aligned[i] and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish alignment, price below 1d EMA50, volume confirmed
            elif (bearish_alignment and close[i] < ema50_1d_aligned[i] and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals