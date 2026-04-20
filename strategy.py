#!/usr/bin/env python3
# 12h_1d_TRIX_SignalLine_Crossover_VolumeConfirm_V1
# Hypothesis: On 12h timeframe, TRIX (12-period) crossing above/below its signal line (9-period EMA of TRIX)
# indicates momentum shifts. Combined with 1d ADX > 25 for trending markets and volume > 1.5x 20-period MA,
# this captures sustained moves. In ranging markets (ADX < 25), we fade at 1d Bollinger Bands (20,2) with volume.
# Targets 15-35 trades/year by requiring TRIX crossover + volume + regime filter.

name = "12h_1d_TRIX_SignalLine_Crossover_VolumeConfirm_V1"
timezone = "UTC"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d TRIX (12-period) and signal line (9-period EMA of TRIX)
    close_1d = df_1d['close'].values
    
    # Triple EMA for TRIX
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix = np.full_like(close_1d, np.nan)
    trix[13:] = 100 * (ema3[13:] - ema3[12:-1]) / ema3[12:-1]
    
    # Signal line = 9-period EMA of TRIX
    signal_line = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 1d Bollinger Bands (20,2) for ranging markets
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Calculate 1d ADX (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align 1d indicators to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    signal_line_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(signal_line_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (ADX > 25): TRIX crossover with volume confirmation
            if adx_aligned[i] > 25:
                # Bullish crossover: TRIX crosses above signal line
                if (trix_aligned[i] > signal_line_aligned[i] and 
                    trix_aligned[i-1] <= signal_line_aligned[i-1] and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Bearish crossover: TRIX crosses below signal line
                elif (trix_aligned[i] < signal_line_aligned[i] and 
                      trix_aligned[i-1] >= signal_line_aligned[i-1] and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
            # Ranging market (ADX < 25): fade at Bollinger Bands with volume
            elif adx_aligned[i] < 25:
                # Long at lower band with volume
                if (close[i] <= lower_bb_aligned[i] * 1.01 and 
                    close[i] >= lower_bb_aligned[i] * 0.99 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Short at upper band with volume
                elif (close[i] >= upper_bb_aligned[i] * 0.99 and 
                      close[i] <= upper_bb_aligned[i] * 1.01 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below signal line or ADX drops to ranging
            if (trix_aligned[i] < signal_line_aligned[i] and 
                trix_aligned[i-1] >= signal_line_aligned[i-1]) or \
               (adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: TRIX crosses above signal line or ADX drops to ranging
            if (trix_aligned[i] > signal_line_aligned[i] and 
                trix_aligned[i-1] <= signal_line_aligned[i-1]) or \
               (adx_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals