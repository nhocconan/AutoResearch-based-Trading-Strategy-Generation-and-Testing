#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKeltnerBreakout_WithVolume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA and ATR calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate weekly EMA(20) of close
    ema_w = pd.Series(close_w).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate weekly ATR(10)
    tr1_w = high_w - low_w
    tr2_w = np.abs(high_w - np.roll(close_w, 1))
    tr3_w = np.abs(low_w - np.roll(close_w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    atr_w = pd.Series(tr_w).rolling(window=10, min_periods=10).mean().values
    
    # Calculate weekly Keltner channels
    upper_w = ema_w + 2.0 * atr_w
    lower_w = ema_w - 2.0 * atr_w
    
    # Align weekly EMA and Keltner channels to daily timeframe
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_w)
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_w)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_w)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120
    
    for i in range(start_idx, n):
        if np.isnan(ema_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema = ema_aligned[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above upper Keltner band + volume
            if price > upper and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Keltner band + volume
            elif price < lower and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below EMA
            if price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above EMA
            if price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals