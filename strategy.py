#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA(50) AND volume > 1.5x avg
# Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA(50) AND volume > 1.5x avg
# Exit when: Alligator lines cross in opposite direction OR price crosses 12h EMA(50)
# Target: 50-150 trades over 4 years (12-37/year) with SMMA smoothing reducing whipsaw

name = "6h_williams_alligator_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def smma(series, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    return pd.Series(series).ewm(alpha=1/period, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components on 6h
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # shift right by 8
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Bearish alignment (Lips < Teeth < Jaw) OR price < 12h EMA(50)
            if (lips[i] < teeth[i] < jaw[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bullish alignment (Lips > Teeth > Jaw) OR price > 12h EMA(50)
            if (lips[i] > teeth[i] > jaw[i]) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + trend filter + volume
            bullish_alignment = (lips[i] > teeth[i] > jaw[i])
            bearish_alignment = (lips[i] < teeth[i] < jaw[i])
            
            if bullish_alignment and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish alignment above 12h EMA - strong uptrend
                signals[i] = 0.25
                position = 1
            elif bearish_alignment and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish alignment below 12h EMA - strong downtrend
                signals[i] = -0.25
                position = -1
    
    return signals