#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot R3/S3 breakout with 4h volume spike and ADX regime filter.
Long when price breaks above 1d Camarilla R3 AND volume > 2.0x 20-period average AND ADX(14) > 25.
Short when price breaks below 1d Camarilla S3 AND volume > 2.0x 20-period average AND ADX(14) > 25.
Exit when price retraces to the 1d Camarilla pivot point (PP) or ATR trailing stop hit (2.5*ATR from highest/lowest since entry).
Uses discrete position sizing (0.30) to control drawdown and fee churn.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Combines structure (Camarilla pivot), momentum (volume spike), and trend filter (ADX) for robustness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3, PP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla levels based on previous day's OHLC
    # For each 1d bar, calculate levels from that bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe (with 1-bar delay for completed bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for regime filter
    # Calculate +DM, -DM, TR
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    # ATR(14) for trailing stop
    atr = wilders_smoothing(tr, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Volume MA needs 20, ADX/ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx[i]
        atr_val = atr[i]
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R3 AND volume spike AND ADX > 25 (trending)
            if (price > r3_val and volume[i] > 2.0 * vol_ma_val and adx_val > 25):
                signals[i] = 0.30
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: Price breaks below 1d Camarilla S3 AND volume spike AND ADX > 25 (trending)
            elif (price < s3_val and volume[i] > 2.0 * vol_ma_val and adx_val > 25):
                signals[i] = -0.30
                position = -1
                entry_price = price
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1d Camarilla pivot point (PP)
            if position == 1 and price <= pp_val:
                exit_signal = True
            elif position == -1 and price >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_4hVolumeSpike_ADX25TrendFilter_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0