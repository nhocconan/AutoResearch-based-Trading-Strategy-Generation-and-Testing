#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA20 on daily close for trend
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate ATR(20) on daily for Keltner channels
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_20_1d = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Calculate 4h EMA20 for Keltner center line
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Bands (2 * ATR multiplier)
    upper_keltner = ema_20_4h + (2 * atr_20_1d_aligned)
    lower_keltner = ema_20_4h - (2 * atr_20_1d_aligned)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_20_1d_aligned[i]) or
            np.isnan(ema_20_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_20_4h_val = ema_20_4h[i]
        upper_keltner_val = upper_keltner[i]
        lower_keltner_val = lower_keltner[i]
        ema_1d_trend_val = ema_20_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Keltner with volume and daily uptrend
            if close_val > upper_keltner_val and vol_filter and (close_val > ema_1d_trend_val):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner with volume and daily downtrend
            elif close_val < lower_keltner_val and vol_filter and (close_val < ema_1d_trend_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20 or volume dries up
            if close_val < ema_20_4h_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 or volume dries up
            if close_val > ema_20_4h_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals