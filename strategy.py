#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w EMA34 trend filter with Camarilla R1/S1 breakout and volume confirmation.
Long when price breaks above 1w Camarilla R1 AND 1w EMA34 is rising AND volume > 1.8x 20-period average.
Short when price breaks below 1w Camarilla S3 AND 1w EMA34 is falling AND volume > 1.8x 20-period average.
Exit when price retouches 1w Camarilla pivot point or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 1d timeframe to target 10-25 trades/year per symbol (40-100 total over 4 years).
Works in both bull and bear markets by trading with the 1w trend and using volume confirmation to filter false breakouts.
1w Camarilla levels provide weekly institutional support/resistance; 1w EMA34 filters counter-trend moves.
"""

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
    open_time = prices['open_time'].values
    
    # Precompute session hours (00-23 UTC) - full day for 1d timeframe
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 1w Camarilla levels (weekly)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels (based on previous week's OHLC)
    camarilla_h1w = (high_1w + low_1w + close_1w) / 3.0
    camarilla_l1w = (high_1w + low_1w + close_1w) / 3.0
    camarilla_range = high_1w - low_1w
    
    camarilla_r1 = camarilla_h1w + camarilla_range * 1.1 / 12.0  # R1 level
    camarilla_s1 = camarilla_l1w - camarilla_range * 1.1 / 12.0  # S1 level
    camarilla_pivot = camarilla_h1w  # Pivot point
    
    # Align Camarilla levels to 1w timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Calculate 1w EMA34 for trend filter
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_1w_34_aligned)
    ema_slope[1:] = ema_1w_34_aligned[1:] - ema_1w_34_aligned[:-1]
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 5, 34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 1w EMA34 rising AND volume spike
            if (price > r1 and 
                ema_slope_val > 0 and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S1 AND 1w EMA34 falling AND volume spike
            elif (price < s1 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pivot:
                exit_signal = True
            elif position == -1 and price >= pivot:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R1S1_1wEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0