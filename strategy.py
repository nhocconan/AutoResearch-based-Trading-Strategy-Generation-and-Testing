# 12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# In bull markets, price breaks above R1 with volume; in bear markets, breaks below S1.
# Volume confirmation filters false breakouts. ATR-based stop loss manages risk.
# Works in both bull and bear by trading breakouts in direction of 12h trend.
# Target: 20-50 trades over 4 years (5-12/year) to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 12h for stop loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    rang = high_1d - low_1d
    camarilla_r1 = close_1d + rang * 1.1 / 12
    camarilla_s1 = close_1d - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 12h EMA(34) for trend filter
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema_34[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_val = ema_34[i]
        
        # Volume filter
        volume_ok = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout above R1 with volume and above EMA34 (uptrend)
            if price > r1 and volume_ok and price > ema_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakdown below S1 with volume and below EMA34 (downtrend)
            elif price < s1 and volume_ok and price < ema_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price drops below entry - 2*ATR (stop loss) or reverses below S1
            if price < entry_price - 2.0 * atr_val or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above entry + 2*ATR (stop loss) or reverses above R1
            if price > entry_price + 2.0 * atr_val or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals