#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d ATR stop
# - Uses 12h Camarilla pivot levels from 1d high/low/close
# - Entry: price breaks above R1 or below S1 + volume > 1.8x 20-period average
# - Stop: 1d ATR(14) * 2 from entry price
# - Trend filter: price above/below 1d EMA50 for long/short bias
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)
# - Works in bull/bear: breakouts capture momentum, volume filter avoids false signals, ATR stop manages risk

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation (pivots need daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 (most important levels for breakouts)
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR for stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + volume surge + above 1d EMA50
            if (price > r1_12h[i] and price > r1_12h[i-1] and 
                vol > 1.8 * vol_ma[i] and price > ema_50_12h[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S1 + volume surge + below 1d EMA50
            elif (price < s1_12h[i] and price < s1_12h[i-1] and 
                  vol > 1.8 * vol_ma[i] and price < ema_50_12h[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below S1 OR ATR stop hit (2*ATR)
            if price < s1_12h[i] or price < entry_price - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR ATR stop hit (2*ATR)
            if price > r1_12h[i] or price > entry_price + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0