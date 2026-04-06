#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation
# Uses daily Camarilla levels (R3/S3 for reversal, R4/S4 for breakout)
# Long at S3 with bullish 1d trend and volume > 1.5x average
# Short at R3 with bearish 1d trend and volume > 1.5x average
# Exit on opposite Camarilla level or trend reversal
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Target: 80-150 trades over 4 years (20-38/year)

name = "6h_camarilla_1d_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1/2)
    # R3 = C + ((H - L) * 1.1/4)
    # S3 = C - ((H - L) * 1.1/4)
    # S4 = C - ((H - L) * 1.1/2)
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1-day EMA(20) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Exit: price reaches R3 or trend turns bearish (below EMA)
            elif close[i] >= r3_1d_aligned[i] or close[i] < ema_1d_aligned[i]:
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
            # Exit: price reaches S3 or trend turns bullish (above EMA)
            elif close[i] <= s3_1d_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at S3/R3 with volume confirmation and trend alignment
            # Long at S3: price touches/below S3, price above EMA (bullish trend), volume spike
            if (close[i] <= s3_1d_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short at R3: price touches/above R3, price below EMA (bearish trend), volume spike
            elif (close[i] >= r3_1d_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals