#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from 1-day timeframe with volume confirmation
# Long when price breaks above R4 with volume > 1.5x average and price > 1d EMA(50)
# Short when price breaks below S4 with volume > 1.5x average and price < 1d EMA(50)
# Exit when price returns to Pivot Point or when EMA trend reverses
# Stoploss at 2.5 * ATR(14)
# Position size: 0.28 (28% of capital)
# Uses 1d EMA to filter counter-trend breakouts and volume to confirm breakout strength
# Target: 60-180 total trades over 4 years (15-45/year)

name = "6h_camarilla_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = pp + (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    r4 = pp + (range_1d * 1.1 / 2)
    # Support levels
    s1 = pp - (range_1d * 1.1 / 12)
    s2 = pp - (range_1d * 1.1 / 6)
    s3 = pp - (range_1d * 1.1 / 4)
    s4 = pp - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Pivot Point or trend turns bearish
            elif close[i] <= pp_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to Pivot Point or trend turns bullish
            elif close[i] >= pp_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above R4, price above EMA (bullish trend), volume spike
            if (close[i] > r4_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4, price below EMA (bearish trend), volume spike
            elif (close[i] < s4_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals