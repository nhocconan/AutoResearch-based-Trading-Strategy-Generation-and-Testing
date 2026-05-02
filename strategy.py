#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator strategy with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trends when lines are aligned and separated
# Long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA(34) + volume spike
# Short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA(34) + volume spike
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring 1d trend alignment and volume confirmation

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator lines and volume MA)
    start_idx = 30  # max(13 for Jaw, 20 for volume MA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: bullish alignment + price > 1d EMA + volume spike
            if bullish_alignment and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price < 1d EMA + volume spike
            elif bearish_alignment and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator lines converge (Teeth crosses below Lips) or price < 1d EMA
            if teeth[i] < lips[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (Teeth crosses above Lips) or price > 1d EMA
            if teeth[i] > lips[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals