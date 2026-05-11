#!/usr/bin/env python3
# 6h_RMI_1dTrend_Volume
# Hypothesis: 6h Relative Momentum Index (RMI) combined with 1d trend filter and volume confirmation.
# Long when: RMI < 30 (oversold), 1d EMA34 rising, volume > 1.8x 20-period average.
# Short when: RMI > 70 (overbought), 1d EMA34 falling, volume > 1.8x 20-period average.
# Exit when RMI crosses 50 or 1d EMA34 trend reverses.
# Works in bull markets by buying dips and in bear by selling rallies with trend filter.
# RMI provides mean-reversion signals, EMA34 filters counter-trend moves, volume confirms strength.

name = "6h_RMI_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- RMI (Relative Momentum Index) ---
    # Calculate RMI(14) - similar to RSI but uses momentum of price changes
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize RMI arrays
    rmi = np.full(n, np.nan)
    avg_up = np.full(n, np.nan)
    avg_down = np.full(n, np.nan)
    
    # Calculate initial average gain/loss
    if len(up) >= period:
        avg_up[period-1] = np.mean(up[:period])
        avg_down[period-1] = np.mean(down[:period])
        
        # Wilder's smoothing
        for i in range(period, n):
            avg_up[i] = (avg_up[i-1] * (period - 1) + up[i]) / period
            avg_down[i] = (avg_down[i-1] * (period - 1) + down[i]) / period
            
            # Calculate RSI then convert to RMI
            if avg_down[i] != 0:
                rs = avg_up[i] / avg_down[i]
                rsi = 100 - (100 / (1 + rs))
                # RMI: map RSI to 0-100 scale with momentum adjustment
                rmi[i] = 50 + (rsi - 50) * 1.5  # Amplify momentum
                # Clamp to 0-100
                rmi[i] = max(0, min(100, rmi[i]))
            else:
                rmi[i] = 100 if avg_up[i] > 0 else 0
    
    # --- 1d EMA34 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # Align 1d EMA and slope to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RMI(14), EMA34, and volume MA(20)
    start_idx = max(30, 34, 20)  # RMI needs ~30 for stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rmi[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RMI conditions
        rmi_oversold = rmi[i] < 30
        rmi_overbought = rmi[i] > 70
        rmi_exit = abs(rmi[i] - 50) < 5  # Exit near center
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.8  # 80% above average
        
        if position == 0:
            if rmi_oversold and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: oversold + rising EMA34 + volume spike
                signals[i] = 0.25
                position = 1
            elif rmi_overbought and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: overbought + falling EMA34 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RMI returns to neutral OR EMA34 slope turns negative
                if rmi_exit or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RMI returns to neutral OR EMA34 slope turns positive
                if rmi_exit or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals