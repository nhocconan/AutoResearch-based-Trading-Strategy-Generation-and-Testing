#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND weekly EMA50 is rising AND volume > 1.8x 20-day average.
Short when price breaks below 20-day low AND weekly EMA50 is falling AND volume > 1.8x 20-day average.
Exit when price retouches 10-day EMA or ATR stoploss hit (1.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 7-25 trades/year per symbol (30-100 total over 4 years) by using weekly trend filter to reduce false breakouts.
Designed to work in both bull and bear markets by trading with the weekly trend and using tight risk control.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate Donchian channels from daily data (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian high/low (based on previous 20 days, not including current)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 10-day EMA for exit
    close_1d = df_1d['close'].values
    ema_10 = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # EMA slope (rising/falling) - compare current vs 2 periods ago (weekly)
    ema_slope = np.zeros_like(ema_1w_50_aligned)
    ema_slope[2:] = ema_1w_50_aligned[2:] - ema_1w_50_aligned[:-2]
    
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
    start_idx = max(100, 20, 10, 50, 20, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(ema_1w_50_aligned[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_exit = ema_10_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly EMA50 rising AND volume spike
            if (price > upper and 
                ema_slope_val > 0 and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian low AND weekly EMA50 falling AND volume spike
            elif (price < lower and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 10-day EMA
            if position == 1 and price < ema_exit:
                exit_signal = True
            elif position == -1 and price > ema_exit:
                exit_signal = True
            
            # ATR-based stoploss: 1.5 * ATR from entry
            if position == 1 and price < entry_price - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_VolumeConfirmation_EMA10Exit_ATRStop"
timeframe = "1d"
leverage = 1.0