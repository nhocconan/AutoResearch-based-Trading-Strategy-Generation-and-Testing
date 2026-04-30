#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Alligator consists of three SMAs (jaw=13, teeth=8, lips=5) shifted forward.
# Long when lips > teeth > jaw (bullish alignment) with 1d uptrend (close > 1d EMA34) and volume > 1.5x 20-bar avg.
# Short when lips < teeth < jaw (bearish alignment) with 1d downtrend (close < 1d EMA34) and volume > 1.5x 20-bar avg.
# Exit when Alligator lines cross (jaws cross lips) or volume drops below threshold.
# Uses proven Williams Alligator for trend identification with strict volume confirmation to limit trades.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_lips = lips.iloc[i]
        curr_teeth = teeth.iloc[i]
        curr_jaw = jaw.iloc[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator alignment (lips > teeth > jaw), uptrend, volume spike
            if (curr_lips > curr_teeth and curr_teeth > curr_jaw and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment (lips < teeth < jaw), downtrend, volume spike
            elif (curr_lips < curr_teeth and curr_teeth < curr_jaw and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Alligator lines cross (jaw crosses lips) or volume drops
            if (curr_jaw >= curr_lips) or (not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Alligator lines cross (jaw crosses lips) or volume drops
            if (curr_jaw <= curr_lips) or (not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals