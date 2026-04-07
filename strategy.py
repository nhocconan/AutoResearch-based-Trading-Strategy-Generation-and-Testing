#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator (Jaw/Teeth/Lips) with 1-day EMA200 trend filter
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA200 (uptrend)
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA200 (downtrend)
# Exit when alignment breaks or price crosses EMA200 in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_williams_alligator_1d_ema200_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams Alligator lines (13, 8, 5 SMAs with shifts)
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks bearish OR price below EMA200
            elif (lips_aligned[i] < teeth_aligned[i] or 
                  teeth_aligned[i] < jaw_aligned[i] or
                  close[i] < ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Alligator alignment breaks bullish OR price above EMA200
            elif (lips_aligned[i] > teeth_aligned[i] or 
                  teeth_aligned[i] > jaw_aligned[i] or
                  close[i] > ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Alligator alignment and trend filter
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = (lips_aligned[i] > teeth_aligned[i] and 
                       teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips_aligned[i] < teeth_aligned[i] and 
                       teeth_aligned[i] < jaw_aligned[i])
            
            # Long: bullish alignment AND price above EMA200 (uptrend)
            if bullish and close[i] > ema_200_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish alignment AND price below EMA200 (downtrend)
            elif bearish and close[i] < ema_200_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals