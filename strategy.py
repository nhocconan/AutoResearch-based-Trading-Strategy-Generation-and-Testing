#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Confirmation + 1d EMA34 Trend Filter
# Williams Alligator uses 3 SMAs (Jaw=13, Teeth=8, Lips=5) to detect trends.
# In trending markets, the SMAs are ordered (Lips > Teeth > Jaw for up, reverse for down).
# Works in both bull and bear markets: long when aligned up, short when aligned down.
# Volume confirmation filters false signals. 1d EMA34 ensures higher timeframe trend alignment.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drift.
name = "12h_WilliamsAlligator_Volume_1dEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components on 12h data
    # Jaw: 13-period SMMA (smoothed moving average)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    # SMMA is similar to EMA but with different smoothing
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume spike: current volume > 1.5 * 24-period average (2 days on 12h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_val = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price above EMA34 AND volume spike
            if lips_val > teeth_val > jaw_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price below EMA34 AND volume spike
            elif lips_val < teeth_val < jaw_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth) or price below EMA34
            if lips_val <= teeth_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth) or price above EMA34
            if lips_val >= teeth_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals