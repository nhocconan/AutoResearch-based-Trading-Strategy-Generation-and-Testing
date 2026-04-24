#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator strategy with 1d trend filter and volume confirmation.
- Primary timeframe: 4h entries/exits based on Williams Alligator (Jaw/Teeth/Lips crossover).
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 1.8 * 20-period 1d volume MA to confirm momentum.
- Entry: Long when Alligator Lips cross above Teeth AND Teeth above Jaw (bullish alignment) 
         AND 1d EMA34 bullish AND volume spike.
         Short when Alligator Lips cross below Teeth AND Teeth below Jaw (bearish alignment)
         AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Alligator crossover or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
- Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3) - smoothed with future shift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 4h
    # Jaw: Blue line - SMA(13, 8) - 13-period SMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: Red line - SMA(8, 5) - 8-period SMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: Green line - SMA(5, 3) - 5-period SMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8)  # Need enough 1d bars for EMA34 and Alligator ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish Alligator: Lips > Teeth > Jaw (green above red above blue)
                bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
                # Bearish Alligator: Lips < Teeth < Jaw (green below red below blue)
                bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
                
                # Bullish entry: bullish alignment AND 1d EMA34 bullish (close > EMA34)
                if bullish_alignment and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: bearish alignment AND 1d EMA34 bearish (close < EMA34)
                elif bearish_alignment and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: bearish Alligator alignment OR loss of volume confirmation
            bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
            if bearish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Alligator alignment OR loss of volume confirmation
            bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
            if bullish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0