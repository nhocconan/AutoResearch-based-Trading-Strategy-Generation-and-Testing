#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD (VMACD) with 1d EMA50 trend filter and ADX25 regime
# VMACD = EMA(volume * close, 12) - EMA(volume * close, 26)
# Long when VMACD crosses above signal line AND price > 1d EMA50 AND ADX > 25
# Short when VMACD crosses below signal line AND price < 1d EMA50 AND ADX > 25
# Uses volume weighting to confirm institutional participation
# Regime filter (ADX > 25) avoids whipsaw in ranging markets
# Target: 12-37 trades/year via strict confluence of volume, momentum, trend and regime

name = "6h_VMACD_1dEMA50_ADX25_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 and ADX calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50 and ADX
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (since we lost first bar in calculations)
    ema_50_1d = np.concatenate([np.full(49, np.nan), ema_50_1d])
    adx = np.concatenate([np.full(27, np.nan), adx])  # 13 (EMA) + 14 (TR) + 14 (ADX smoothing) - 1
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Volume-Weighted MACD on 6h data
    # VMACD = EMA(volume * close, 12) - EMA(volume * close, 26)
    vc = volume * close  # volume-weighted close
    vc_series = pd.Series(vc)
    ema_vc_12 = vc_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_vc_26 = vc_series.ewm(span=26, adjust=False, min_periods=26).mean().values
    vmacd = ema_vc_12 - ema_vc_26
    
    # Signal line: EMA of VMACD
    vmacd_series = pd.Series(vmacd)
    signal_line = vmacd_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # VMACD histogram (for crossover detection)
    vmacd_hist = vmacd - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26, 9)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vmacd[i]) or np.isnan(signal_line[i]) or np.isnan(vmacd_hist[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        price = close[i]
        hist = vmacd_hist[i]
        hist_prev = vmacd_hist[i-1]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when VMACD crosses above zero (bullish momentum) AND price > 1d EMA50 AND ADX > 25
            if hist_prev <= 0 and hist > 0 and price > ema_50_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short when VMACD crosses below zero (bearish momentum) AND price < 1d EMA50 AND ADX > 25
            elif hist_prev >= 0 and hist < 0 and price < ema_50_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when VMACD crosses below zero or ADX < 20 (range)
            if hist >= 0 and hist_prev > 0 and hist < hist_prev:  # momentum weakening
                signals[i] = 0.0
                position = 0
            elif adx_val < 20:  # regime change to ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when VMACD crosses above zero or ADX < 20 (range)
            if hist >= 0 and hist_prev < 0 and hist > hist_prev:  # momentum weakening
                signals[i] = 0.0
                position = 0
            elif adx_val < 20:  # regime change to ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals