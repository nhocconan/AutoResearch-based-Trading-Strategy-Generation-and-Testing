#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when: price breaks above R1, close > EMA34(1d), volume > 1.5x 20-period average
# Short when: price breaks below S1, close < EMA34(1d), volume > 1.5x 20-period average
# Exit when: price returns to opposite pivot level (S1 for longs, R1 for shorts)
# Camarilla levels from daily OHLC provide institutional support/resistance.
# EMA34 filters trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakout strength.
# Target: 15-25 trades/year per symbol. Works in bull (buy R1 breaks) and bear (sell S1 breaks).
name = "12h_Camarilla_R1S1_Breakout_EMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L)*1.12/12, S1 = C - (H-L)*1.12/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12 / 12
    
    # Calculate EMA34 on daily close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily data to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above R1, close > EMA34, volume confirmation
            if (price > r1 and close[i] > ema34 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, close < EMA34, volume confirmation
            elif (price < s1 and close[i] < ema34 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below S1 (opposite support level)
            if price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above R1 (opposite resistance level)
            if price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals