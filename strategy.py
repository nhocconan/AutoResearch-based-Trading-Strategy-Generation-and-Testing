#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_And_Trend_v1"
timeframe = "4h"
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
    
    # Calculate 1d ATR(14) for Camarilla width
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels using previous day's data
    # Camarilla: H4 = C + 1.1/2 * (H-L), L4 = C - 1.1/2 * (H-L)
    # We use shifted values to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_H4 = prev_close + 1.1/2 * (prev_high - prev_low)
    camarilla_L4 = prev_close - 1.1/2 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # 4h trend filter: EMA(34) slope
    close_s = pd.Series(close)
    ema_34 = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34 - np.roll(ema_34, 1)
    ema_34_slope[0] = 0
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or \
           np.isnan(ema_34_slope[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: bullish when EMA slope > 0
        bullish_trend = ema_34_slope[i] > 0
        bearish_trend = ema_34_slope[i] < 0
        
        if position == 0:
            # Long: price breaks above H4 with volume and bullish trend
            if price > camarilla_H4_aligned[i] and volume_ok and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 with volume and bearish trend
            elif price < camarilla_L4_aligned[i] and volume_ok and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below H4 or trend turns bearish
            if price < camarilla_H4_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above L4 or trend turns bullish
            if price > camarilla_L4_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals