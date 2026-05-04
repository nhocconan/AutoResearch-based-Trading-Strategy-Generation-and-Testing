#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Alligator (Jaw=13, Teeth=8, Lips=5) to identify trends: long when Lips > Teeth > Jaw and price > Teeth,
# short when Lips < Teeth < Jaw and price < Teeth. 1d EMA34 filter ensures alignment with higher-timeframe trend.
# Volume spike confirmation (>1.5x 20-period EMA) filters low-momentum breakouts.
# Discrete position sizing (0.25) minimizes fee churn. Target: 12-25 trades/year per symbol.
# Designed to work in both bull and bear markets by following the 1d trend while using Alligator for entry timing.

name = "6h_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs with specific periods and shifts
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    # Using EMA as approximation for SMMA (common practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    jaw_aligned = jaw.values
    teeth_aligned = teeth.values
    lips_aligned = lips.values
    
    # Get 6h data for volume EMA(20) for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h volume EMA(20) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ema_20 = pd.Series(vol_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1d_aligned[i]
        bearish_trend = close[i] < ema_34_1d_aligned[i]
        
        # Alligator conditions
        # Bullish: Lips > Teeth > Jaw (alligator mouth opening up)
        # Bearish: Lips < Teeth < Jaw (alligator mouth opening down)
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: bullish alligator + price > teeth + volume confirmation + bullish 1d trend
            if (bullish_alligator and close[i] > teeth_aligned[i] and 
                volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: bearish alligator + price < teeth + volume confirmation + bearish 1d trend
            elif (bearish_alligator and close[i] < teeth_aligned[i] and 
                  volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alligator OR price < teeth OR bearish 1d trend
            if (bearish_alligator or close[i] < teeth_aligned[i] or bearish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alligator OR price > teeth OR bullish 1d trend
            if (bullish_alligator or close[i] > teeth_aligned[i] or bullish_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals