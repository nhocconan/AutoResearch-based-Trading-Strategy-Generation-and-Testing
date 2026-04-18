#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d volume confirmation and 1w trend filter.
# Long when: Jaw < Teeth < Lips (bullish alignment), price > Lips, 1d volume > 1.5x 20-day average, 1w close > 1w EMA(34)
# Short when: Jaw > Teeth > Lips (bearish alignment), price < Lips, 1d volume > 1.5x 20-day average, 1w close < 1w EMA(34)
# Exit when Alligator alignment breaks or price crosses Jaw.
# Uses Alligator for trend, volume for confirmation, higher timeframe for trend filter.
# Designed for ~20-30 trades/year per symbol.
name = "4h_WilliamsAlligator_Volume_1wTrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Williams Alligator (13,8,5 SMAs shifted)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(close_4h).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(close_4h).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.shift(3)
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips.values)
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / (volume_ma_20 + 1e-10)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i]) or np.isnan(ema_1w_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        volume_ratio_val = volume_ratio_aligned[i]
        ema_1w = ema_1w_34_aligned[i]
        
        # Bullish alignment: Jaw < Teeth < Lips
        bullish_alignment = jaw_val < teeth_val < lips_val
        # Bearish alignment: Jaw > Teeth > Lips
        bearish_alignment = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: Bullish alignment, price above Lips, volume confirmation, 1w uptrend
            if (bullish_alignment and price > lips_val and 
                volume_ratio_val > 1.5 and close[i] > ema_1w):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, price below Lips, volume confirmation, 1w downtrend
            elif (bearish_alignment and price < lips_val and 
                  volume_ratio_val > 1.5 and close[i] < ema_1w):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alignment breaks or price crosses Jaw
            if not bullish_alignment or price < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alignment breaks or price crosses Jaw
            if not bearish_alignment or price > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals