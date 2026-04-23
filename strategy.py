#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h/1d Camarilla R3/S3 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 4h Camarilla R3 AND volume > 1.5x 20-period average.
Short when price breaks below 4h Camarilla S3 AND volume > 1.5x 20-period average.
Exit when price retraces to 4h Camarilla midpoint or ATR stoploss hit (1.5*ATR).
Uses discrete position sizing (0.20) to minimize fee churn and manage drawdown.
Designed for 1h timeframe to target 15-37 trades/year per symbol (60-150 total over 4 years).
Works in both bull and bear markets by using HTF Camarilla levels for structure and volume to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    # Group 4h bars by date to get daily OHLC
    df_4h_copy = df_4h.copy()
    df_4h_copy['date'] = df_4h_copy.index.date
    daily_ohlc = df_4h_copy.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })
    
    if len(daily_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_daily = daily_ohlc['high'].values
    low_daily = daily_ohlc['low'].values
    close_daily = daily_ohlc['close'].values
    
    # Camarilla R3, S3, and midpoint
    camarilla_r3 = close_daily + 1.1 * (high_daily - low_daily) / 2
    camarilla_s3 = close_daily - 1.1 * (high_daily - low_daily) / 2
    camarilla_mid = close_daily
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_mid = np.roll(camarilla_mid, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    camarilla_mid[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(10) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        mid = camarilla_mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h Camarilla R3 AND volume spike
            if (price > r3 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below 4h Camarilla S3 AND volume spike
            elif (price < s3 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 4h Camarilla midpoint
            if position == 1 and price <= mid:
                exit_signal = True
            elif position == -1 and price >= mid:
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_VolumeConfirmation_ATRStop"
timeframe = "1h"
leverage = 1.0