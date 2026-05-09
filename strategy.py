#!/usr/bin/env python3
# 1d_Trix_Signal_Line_Cross_With_Volume_Filter
# Hypothesis: TRIX signal line crossovers on 1d timeframe with volume confirmation and 1w trend filter.
# TRIX filters out market noise and identifies momentum changes. Works in bull/bear markets
# by using 1w EMA trend filter to avoid counter-trend trades. Volume spike confirms institutional
# participation in the breakout. Target: 20-50 trades over 4 years.

name = "1d_Trix_Signal_Line_Cross_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX on 1d timeframe (15-period EMA triple smoothed)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Triple EMA: EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = np.full_like(close_1d, np.nan)
    ema2 = np.full_like(close_1d, np.nan)
    ema3 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 15:
        ema1[14] = np.mean(close_1d[0:15])
        for i in range(15, len(close_1d)):
            ema1[i] = (close_1d[i] * 2 + ema1[i-1] * 14) / 15
        
        valid_ema1 = ~np.isnan(ema1)
        if np.any(valid_ema1):
            first_valid = np.where(valid_ema1)[0][0]
            ema2[first_valid + 14] = np.mean(ema1[first_valid:first_valid+15])
            for i in range(first_valid + 15, len(close_1d)):
                if not np.isnan(ema1[i]):
                    ema2[i] = (ema1[i] * 2 + ema2[i-1] * 14) / 15
            
            valid_ema2 = ~np.isnan(ema2)
            if np.any(valid_ema2):
                first_valid2 = np.where(valid_ema2)[0][0]
                ema3[first_valid2 + 14] = np.mean(ema2[first_valid2:first_valid2+15])
                for i in range(first_valid2 + 15, len(close_1d)):
                    if not np.isnan(ema2[i]):
                        ema3[i] = (ema2[i] * 2 + ema3[i-1] * 14) / 15
    
    # TRIX: percentage change of triple EMA
    trix = np.full_like(close_1d, np.nan)
    valid_ema3 = ~np.isnan(ema3)
    if np.sum(valid_ema3) > 1:
        trix[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # TRIX signal line: 9-period EMA of TRIX
    trix_signal = np.full_like(trix, np.nan)
    valid_trix = ~np.isnan(trix)
    if np.sum(valid_trix) > 9:
        first_valid_trix = np.where(valid_trix)[0][0]
        trix_signal[first_valid_trix + 8] = np.mean(trix[first_valid_trix:first_valid_trix+9])
        for i in range(first_valid_trix + 9, len(trix)):
            if not np.isnan(trix[i]):
                trix_signal[i] = (trix[i] * 2 + trix_signal[i-1] * 8) / 9
    
    # Align TRIX and signal line to 1d timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 50
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (volume[i] * 2 + vol_ma[i-1] * 19) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above signal line AND uptrend (price > EMA50) AND volume spike
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal line AND downtrend (price < EMA50) AND volume spike
            elif (trix_aligned[i] < trix_signal_aligned[i] and 
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line OR trend reversal (price < EMA50)
            if (trix_aligned[i] < trix_signal_aligned[i] and 
                trix_aligned[i-1] >= trix_signal_aligned[i-1]) or \
               close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line OR trend reversal (price > EMA50)
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1]) or \
               close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals