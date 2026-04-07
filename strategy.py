#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination for trend filtering
# Long when: ADX > 25 (strong trend), price > Alligator Jaw (EMA13), and Alligator Mouth opens upward (Teeth > Lips)
# Short when: ADX > 25, price < Alligator Jaw, and Alligator Mouth opens downward (Teeth < Lips)
# Exit when ADX < 20 (weak trend) or opposite conditions met
# Uses 1d ADX and Alligator for trend direction, 6h for entry timing
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for ADX and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[up_move < 0] = 0
    down_move[down_move < 0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm14 = pd.Series(up_move).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm14 = pd.Series(down_move).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx[np.isnan(adx)] = 0
    
    # 1d Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, center=False).mean().values
    jaw = np.roll(jaw, 8)
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, center=False).mean().values
    teeth = np.roll(teeth, 5)
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, center=False).mean().values
    lips = np.roll(lips, 3)
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ATR(14) for stoploss on 6h
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr2_6h[0] = tr1_6h[0]
    tr3_6h[0] = tr1_6h[0]
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: weak trend (ADX < 20) or Alligator closes (Teeth < Lips)
            elif adx_aligned[i] < 20 or teeth_aligned[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr_6h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: weak trend (ADX < 20) or Alligator closes (Teeth > Lips)
            elif adx_aligned[i] < 20 or teeth_aligned[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with strong trend and Alligator alignment
            # Long: ADX > 25, price above Jaw, Mouth opens up (Teeth > Lips)
            if (adx_aligned[i] > 25 and
                close[i] > jaw_aligned[i] and
                teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: ADX > 25, price below Jaw, Mouth opens down (Teeth < Lips)
            elif (adx_aligned[i] > 25 and
                  close[i] < jaw_aligned[i] and
                  teeth_aligned[i] < lips_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals