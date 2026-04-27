#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily ATR-based breakout levels
    atr_mult = 0.5
    upper_break = df_1d['close'].shift(1) + atr14 * atr_mult
    lower_break = df_1d['close'].shift(1) - atr14 * atr_mult
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly EMA34 and daily ATR
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1w_aligned[i]
        atr_val = atr14_aligned[i]
        upper_break_val = upper_break_aligned[i]
        lower_break_val = lower_break_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_filter = vol_current > (vol_ma_val * 1.3)
        
        if position == 0:
            # Long: price breaks above upper level with weekly uptrend and volume
            if close[i] > upper_break_val and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower level with weekly downtrend and volume
            elif close[i] < lower_break_val and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA or ATR-based trailing stop
            if close[i] < ema_trend or close[i] < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above EMA or ATR-based trailing stop
            if close[i] > ema_trend or close[i] > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ATR_Breakout_EMA34Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0