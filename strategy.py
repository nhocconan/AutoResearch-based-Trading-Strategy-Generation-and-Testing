#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R4 AND 1w EMA34 is rising AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S4 AND 1w EMA34 is falling AND volume > 1.5x 20-period average.
Exit when price retouches Camarilla pivot point (PP) or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 7-25 trades/year per symbol (30-100 total over 4 years) by using 1w trend filter to reduce false breakouts.
Designed to work in both bull and bear markets by trading with the 1w trend and using tight risk control.
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
    
    # Calculate Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = camarilla_pp + (high_1d - low_1d) * 1.1 / 2.0  # R4 level
    camarilla_s4 = camarilla_pp - (high_1d - low_1d) * 1.1 / 2.0  # S4 level
    
    # Align Camarilla levels to 1d timeframe (prices index)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # EMA slope (rising/falling) - compare current vs 2 periods ago (for weekly)
    ema_slope = np.zeros_like(ema_1w_34_aligned)
    ema_slope[2:] = ema_1w_34_aligned[2:] - ema_1w_34_aligned[:-2]
    
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
    start_idx = max(34, 20, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        pp = camarilla_pp_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R4 AND 1w EMA34 rising AND volume confirmation
            if (price > r4 and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S4 AND 1w EMA34 falling AND volume confirmation
            elif (price < s4 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
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

name = "1D_Camarilla_R4S4_1wEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "1d"
leverage = 1.0