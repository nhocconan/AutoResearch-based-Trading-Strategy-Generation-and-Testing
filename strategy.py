#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Daily data for Camarilla levels (calculated from previous day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rng = prev_high - prev_low
    camarilla_r1 = prev_close + rng * 1.1 / 12
    camarilla_s1 = prev_close - rng * 1.1 / 12
    
    # Align to 1h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h trend filter (EMA34) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === 1h price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough warmup for indicators
        # Skip if any critical value is NaN
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 with volume, in uptrend (price > EMA34)
            if (close_val > camarilla_r1_aligned[i] and 
                vol_ratio_val > 1.5 and 
                close_val > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 with volume, in downtrend (price < EMA34)
            elif (close_val < camarilla_s1_aligned[i] and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Camarilla S1 or trend turns down
            if (close_val < camarilla_s1_aligned[i] or 
                close_val < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price breaks above Camarilla R1 or trend turns up
            if (close_val > camarilla_r1_aligned[i] or 
                close_val > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals