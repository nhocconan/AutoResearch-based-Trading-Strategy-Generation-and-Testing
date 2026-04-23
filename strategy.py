#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla R1/S1 breakout with 4h HMA21 trend filter and volume confirmation.
Long when price breaks above 1d Camarilla R1 AND 4h HMA21 is rising AND volume > 1.8x 30-period average.
Short when price breaks below 1d Camarilla S1 AND 4h HMA21 is falling AND volume > 1.8x 30-period average.
Exit when price retraces 50% of the breakout move or ATR stoploss hit (1.5*ATR).
Uses discrete position sizing (0.25) to balance reward and risk while minimizing fee churn.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Works in both bull and bear markets by trading with the 4h trend and using volume confirmation to filter false breakouts.
1d Camarilla levels provide strong institutional support/resistance; 4h HMA21 filters noise and lag.
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
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_h1 = (high_1d + low_1d + close_1d) / 3.0
    camarilla_l1 = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    
    camarilla_r1 = camarilla_h1 + camarilla_range * 1.1 / 12.0
    camarilla_s1 = camarilla_l1 - camarilla_range * 1.1 / 12.0
    camarilla_pivot = camarilla_h1  # Pivot point
    
    # Align Camarilla levels to 1d timeframe (then to 4h via align_htf_to_ltf)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate 4h HMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    half_n = 21 // 2
    wma_half = np.array([wma(close_4h[i-half_n+1:i+1], half_n) if i >= half_n-1 else np.nan for i in range(len(close_4h))])
    wma_full = np.array([wma(close_4h[i-21+1:i+1], 21) if i >= 20 else np.nan for i in range(len(close_4h))])
    hma_4h = 2 * wma_half - wma_full
    hma_4h = np.array([wma(hma_4h[i-int(np.sqrt(21))+1:i+1], int(np.sqrt(21))) if i >= int(np.sqrt(21))-1 else np.nan for i in range(len(hma_4h))])
    
    # Align HMA to 4h timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # HMA slope (rising/falling)
    hma_slope = np.zeros_like(hma_4h_aligned)
    hma_slope[1:] = hma_4h_aligned[1:] - hma_4h_aligned[:-1]
    
    # Volume average (30-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
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
    breakout_level = 0.0  # Track breakout level for 50% retracement exit
    
    # Start from index where all indicators are ready
    start_idx = max(100, 5, 21, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(hma_4h_aligned[i]) or 
            np.isnan(hma_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                breakout_level = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        hma_slope_val = hma_slope[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R1 AND 4h HMA21 rising AND volume spike
            if (price > r1 and 
                hma_slope_val > 0 and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                breakout_level = r1  # Breakout level for exit calculation
            # Short: Price breaks below 1d Camarilla S1 AND 4h HMA21 falling AND volume spike
            elif (price < s1 and 
                  hma_slope_val < 0 and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                breakout_level = s1  # Breakout level for exit calculation
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces 50% of the breakout move
            if position == 1:
                retracement_level = breakout_level + 0.5 * (price - breakout_level)  # This is wrong, let me fix
                # Correct: 50% retracement from breakout level toward pivot
                retracement_level = breakout_level + 0.5 * (pivot - breakout_level)
                if price <= retracement_level:
                    exit_signal = True
            elif position == -1:
                # 50% retracement from breakout level toward pivot
                retracement_level = breakout_level + 0.5 * (pivot - breakout_level)
                if price >= retracement_level:
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
                breakout_level = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_4hHMA21_Trend_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0